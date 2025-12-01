"""
Enhanced FastAPI Backend for RAG Demo
Features:
- Hybrid Search (Dense + Sparse BM25 with RRF fusion)
- LLM-based Reranking
- Deduplication
- Hierarchical context
Using BGE-M3 for embeddings and Qwen3:14b for LLM
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os
from pathlib import Path
import json
import re
import math
from collections import defaultdict, Counter
import google.generativeai as genai
from hierarchical_chunking import HierarchicalChunker

app = FastAPI(title="RAG Demo API - Enhanced")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    EMBEDDING_MODEL = "bge-m3"
    LLM_MODEL = "qwen3:14b"

    # Gemini API configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDigZIj90ppAcue0sWm85ODzB6KIu62ub8")
    GEMINI_MODEL = "gemini-2.0-flash"  # Fast and cheap for chunking

    # Feature flags
    USE_HYBRID_SEARCH = True
    USE_RERANKING = False  # Disabled for faster query
    USE_DEDUPLICATION = True
    USE_SEMANTIC_CHUNKING = False  # DISABLED - Causes information loss
    USE_CONTEXT_ENRICHMENT = True  # Add parent and related chunks to context
    USE_HIERARCHICAL_CHUNKING = True  # Use section-based hierarchical chunking for DOCX

    # Parameters
    RERANKING_MULTIPLIER = 3
    DEDUP_THRESHOLD = 0.7

    # BM25 parameters
    BM25_K1 = 1.5
    BM25_B = 0.75

    # RRF parameter
    RRF_K = 60

    # Chunking parameters
    MIN_CHUNK_SIZE = 10     # Minimum words per chunk (lowered to keep small sections)
    TARGET_CHUNK_SIZE = 200  # Target words per chunk (smaller for better search)
    MAX_CHUNK_SIZE = 400     # Maximum words per chunk (reduced from 800 for finer granularity)
    CHUNK_OVERLAP = 30       # Overlap between chunks in words
    SEMANTIC_SPLIT_THRESHOLD = 300  # If chunk > this, try semantic split

config = Config()

# Initialize Gemini
if config.GEMINI_API_KEY:
    genai.configure(api_key=config.GEMINI_API_KEY)
    print(f"✓ Gemini API configured with model: {config.GEMINI_MODEL}")
else:
    print("⚠ GEMINI_API_KEY not set - semantic chunking will be disabled")

# Initialize Hierarchical Chunker
hierarchical_chunker = HierarchicalChunker(gemini_api_key=config.GEMINI_API_KEY)
print(f"✓ Hierarchical chunker initialized")

# In-memory storage
vector_store = {
    "documents": [],
    "metadata": []
}


class QueryRequest(BaseModel):
    query: str
    topK: int = 3


class ContextChunk(BaseModel):
    type: str  # Always "primary" for frontend compatibility
    heading: str
    headingPath: str
    content: str
    similarity: Optional[float] = None
    importance: Optional[float] = None
    level: Optional[int] = None  # Hierarchy level
    contextType: Optional[str] = None  # "match", "parent", or "related" (for backend tracking)
    relatedTo: Optional[str] = None  # Which primary chunk this is related to
    relationshipType: Optional[str] = None  # sibling, child (for related chunks)


class QueryResponse(BaseModel):
    answer: str
    retrievedDocuments: List[Dict[str, Any]]
    contextStructure: Dict[str, Any]


# BM25 Implementation
class BM25:
    """BM25 sparse retrieval implementation"""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        # Remove punctuation, lowercase, split
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [word for word in cleaned.split() if len(word) > 1]
        return tokens

    @staticmethod
    def calculate_idf(query_terms: List[str], documents: List[str]) -> Dict[str, float]:
        """Calculate IDF scores"""
        idf_scores = {}
        total_docs = len(documents)

        for term in query_terms:
            docs_with_term = sum(1 for doc in documents if term in BM25.tokenize(doc))
            # IDF formula with smoothing
            idf = math.log((total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1.0)
            idf_scores[term] = idf

        return idf_scores

    @staticmethod
    def calculate_bm25_scores(query: str, documents: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
        """Calculate BM25 scores for all documents"""
        query_terms = BM25.tokenize(query)
        if not query_terms:
            return [0.0] * len(documents)

        # Calculate average document length
        doc_lengths = [len(BM25.tokenize(doc)) for doc in documents]
        avg_doc_length = sum(doc_lengths) / len(documents) if documents else 0

        # Calculate IDF
        idf_scores = BM25.calculate_idf(query_terms, documents)

        # Calculate BM25 score for each document
        scores = []
        for doc, doc_len in zip(documents, doc_lengths):
            doc_terms = BM25.tokenize(doc)
            term_freq = Counter(doc_terms)

            score = 0.0
            for term in query_terms:
                if term in term_freq:
                    tf = term_freq[term]
                    idf = idf_scores[term]

                    # BM25 formula
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_length)
                    score += idf * (numerator / denominator)

            scores.append(score)

        return scores


def reciprocal_rank_fusion(dense_scores: List[float], sparse_scores: List[float], k: int = 60) -> List[float]:
    """
    Reciprocal Rank Fusion (RRF) to combine dense and sparse retrieval
    RRF(d) = Σ 1/(k + rank(d))
    """
    # Get rankings (indices sorted by score descending)
    dense_ranks = np.argsort(dense_scores)[::-1]
    sparse_ranks = np.argsort(sparse_scores)[::-1]

    # Create rank position maps
    dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranks)}
    sparse_rank_map = {idx: rank for rank, idx in enumerate(sparse_ranks)}

    # Calculate RRF scores
    rrf_scores = []
    for i in range(len(dense_scores)):
        dense_rank = dense_rank_map.get(i, len(dense_scores))
        sparse_rank = sparse_rank_map.get(i, len(sparse_scores))

        rrf_score = (1.0 / (k + dense_rank)) + (1.0 / (k + sparse_rank))
        rrf_scores.append(rrf_score)

    return rrf_scores


def deduplicate_documents(docs: List[Dict], threshold: float = 0.7) -> List[Dict]:
    """Remove duplicate documents using Jaccard similarity"""
    unique_docs = []

    for doc in docs:
        is_duplicate = False
        doc_words = set(doc['content'].lower().split())

        for unique_doc in unique_docs:
            unique_words = set(unique_doc['content'].lower().split())

            # Jaccard similarity
            intersection = doc_words & unique_words
            union = doc_words | unique_words

            if union:
                jaccard = len(intersection) / len(union)
                if jaccard > threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_docs.append(doc)

    return unique_docs


def rerank_documents(query: str, documents: List[Dict], topK: int) -> List[Dict]:
    """Rerank documents using LLM scoring"""
    print(f"Reranking {len(documents)} documents...")

    scored_docs = []
    for i, doc in enumerate(documents):
        # Truncate content if too long
        content = doc['content'][:1000] + "..." if len(doc['content']) > 1000 else doc['content']

        # Score relevance
        score = score_relevance(query, content, i + 1, len(documents))
        scored_docs.append({**doc, 'rerank_score': score})
        print(f"  Document {i+1}: Relevance score = {score:.2f}")

    # Sort by rerank score
    scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

    return scored_docs[:topK]


def score_relevance(query: str, content: str, doc_index: int, total_docs: int) -> float:
    """Score document relevance using LLM (0-10 scale)"""
    try:
        prompt = f"""You are a relevance scoring assistant for a RAG system.

Task: Score how relevant the following document is to answer the user's question.

Question: {query}

Document:
{content}

Instructions:
- Score from 0 to 10 (0 = completely irrelevant, 10 = highly relevant)
- Consider: Does this document contain information to answer the question?
- Be strict: Only give high scores (8-10) if the document directly answers the question
- Respond with ONLY a number (0-10), no explanation needed

Score:"""

        response = ollama.generate(
            model=config.LLM_MODEL,
            prompt=prompt,
            options={
                "temperature": 0.1,
                "num_predict": 5
            }
        )

        # Extract score
        answer = response['response'].strip()
        score_str = re.sub(r'[^0-9.]', '', answer)
        if score_str:
            score = float(score_str)
            return max(0.0, min(10.0, score))

    except Exception as e:
        print(f"Error scoring document {doc_index}/{total_docs}: {e}")

    return 5.0  # Default medium score


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from uploaded file"""
    ext = Path(filename).suffix.lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    elif ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex"""
    # Split on common sentence endings
    sentences = re.split(r'([.!?]\s+|\n\n)', text)
    # Reconstruct sentences with their punctuation
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            result.append(sentences[i] + sentences[i+1].strip())
        else:
            result.append(sentences[i])
    if len(sentences) % 2 == 1:
        result.append(sentences[-1])
    return [s.strip() for s in result if s.strip()]


def semantic_split_with_llm(text: str, heading: str = "") -> List[str]:
    """
    Use Gemini API to intelligently split text into semantic chunks.
    Returns list of text chunks with natural boundaries.
    """
    if not config.GEMINI_API_KEY:
        # Fallback to sentence-based splitting
        return sentence_based_split(text, config.TARGET_CHUNK_SIZE, config.CHUNK_OVERLAP)

    word_count = len(text.split())

    # If text is small enough, return as-is
    if word_count <= config.MAX_CHUNK_SIZE:
        return [text]

    try:
        # Calculate target number of chunks
        num_chunks = max(2, math.ceil(word_count / config.TARGET_CHUNK_SIZE))

        prompt = f"""You are a text chunking assistant. Your task is to split the following text into {num_chunks} meaningful chunks.

Guidelines:
- Each chunk should be semantically coherent (complete ideas, not cut mid-sentence)
- Aim for roughly {config.TARGET_CHUNK_SIZE} words per chunk
- Split at natural boundaries (paragraph breaks, topic shifts, etc.)
- Preserve all original text - don't summarize or modify
- Output ONLY the split points as line numbers (where to split)

Text to split:
{text[:8000]}{"..." if len(text) > 8000 else ""}

Respond with split points as a JSON array of numbers representing character positions.
Example: [500, 1200, 1800] means split at positions 500, 1200, and 1800.

Split points:"""

        model = genai.GenerativeModel(config.GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 500
            }
        )

        # Parse response
        answer = response.text.strip()
        # Extract JSON array
        json_match = re.search(r'\[[\d,\s]+\]', answer)
        if json_match:
            split_points = json.loads(json_match.group())

            # Split text at these points
            chunks = []
            prev_pos = 0
            for pos in split_points:
                if pos > prev_pos and pos < len(text):
                    chunks.append(text[prev_pos:pos].strip())
                    prev_pos = pos
            # Add remaining text
            if prev_pos < len(text):
                chunks.append(text[prev_pos:].strip())

            # Filter out empty chunks
            chunks = [c for c in chunks if c]
            if chunks:
                print(f"  ✓ Semantic split: {len(chunks)} chunks using Gemini")
                return chunks

    except Exception as e:
        print(f"  ⚠ Gemini semantic split failed: {e}, falling back to sentence-based")

    # Fallback to sentence-based splitting
    return sentence_based_split(text, config.TARGET_CHUNK_SIZE, config.CHUNK_OVERLAP)


def sentence_based_split(text: str, target_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks based on sentences, maintaining semantic coherence.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [text]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # If adding this sentence exceeds target, start new chunk
        if current_word_count + sentence_words > target_size and current_chunk:
            chunks.append(' '.join(current_chunk))

            # Create overlap by keeping last few sentences
            overlap_sentences = []
            overlap_words = 0
            for sent in reversed(current_chunk):
                sent_words = len(sent.split())
                if overlap_words + sent_words <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_words += sent_words
                else:
                    break

            current_chunk = overlap_sentences
            current_word_count = overlap_words

        current_chunk.append(sentence)
        current_word_count += sentence_words

    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def recursive_split_large_chunks(text: str, heading: str = "", max_size: int = None) -> List[str]:
    """
    Recursively split text if it exceeds max_size.
    Uses semantic splitting with LLM when available.
    """
    if max_size is None:
        max_size = config.MAX_CHUNK_SIZE

    word_count = len(text.split())

    # Base case: text is small enough
    if word_count <= max_size:
        return [text]

    print(f"  → Large chunk detected ({word_count} words), splitting...")

    # Try semantic split if enabled
    if config.USE_SEMANTIC_CHUNKING and word_count >= config.SEMANTIC_SPLIT_THRESHOLD:
        chunks = semantic_split_with_llm(text, heading)
    else:
        # Use sentence-based split
        chunks = sentence_based_split(text, config.TARGET_CHUNK_SIZE, config.CHUNK_OVERLAP)

    # Recursively split any chunks that are still too large
    final_chunks = []
    for chunk in chunks:
        if len(chunk.split()) > max_size:
            final_chunks.extend(recursive_split_large_chunks(chunk, heading, max_size))
        else:
            final_chunks.append(chunk)

    return final_chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks (fallback method).
    Now uses sentence-aware splitting for better coherence.
    """
    return sentence_based_split(text, chunk_size, overlap)


def get_embedding(text: str) -> np.ndarray:
    """Get embedding vector from Ollama BGE-M3"""
    try:
        response = ollama.embed(model=config.EMBEDDING_MODEL, input=text)
        return np.array(response['embeddings'][0])
    except Exception as e:
        print(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")


def generate_answer(query: str, context_chunks: List[str], metadata_list: List[Dict[str, Any]] = None) -> str:
    """
    Generate answer using Qwen3:14b with professional prompt.
    Now includes rich metadata from Excel in context.
    """
    # Build context with metadata if available
    if metadata_list:
        context_parts = []
        for i, (chunk, meta) in enumerate(zip(context_chunks, metadata_list)):
            # Extract useful metadata
            heading_path = meta.get("heading_path", "")
            section_code = meta.get("section_code", "")
            tags = meta.get("tags_raw", "")
            business_module = meta.get("business_module", "")

            # Build context block with metadata
            context_block = f"Tài liệu {i+1}"
            if heading_path:
                context_block += f" - Đường dẫn: {heading_path}"
            if section_code:
                context_block += f" [{section_code}]"
            if tags:
                context_block += f"\nTags: {tags}"
            if business_module:
                context_block += f"\nModule: {business_module}"
            context_block += f"\n\nNội dung:\n{chunk}"

            context_parts.append(context_block)

        context = "\n\n---\n\n".join(context_parts)
    else:
        # Fallback to simple format
        context = "\n\n---\n\n".join([f"Tài liệu {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])

    prompt = f"""Bạn là trợ lý ảo hỗ trợ chăm sóc khách hàng của phần mềm Tendoo - phần mềm quản lý bán hàng.

Nhiệm vụ của bạn:
1. Trả lời câu hỏi của khách hàng một cách chính xác, rõ ràng và thân thiện
2. Nếu câu hỏi yêu cầu hướng dẫn thực hiện một tác vụ, hãy cung cấp các bước chi tiết
3. Sử dụng bullet points hoặc đánh số để trình bày các bước một cách dễ hiểu
4. Nếu có nhiều cách thực hiện, hãy đề xuất cách đơn giản và phổ biến nhất
5. Luôn trả lời bằng tiếng Việt với giọng điệu chuyên nghiệp nhưng thân thiện

Hướng dẫn trả lời QUAN TRỌNG:
- Đọc kỹ Context bên dưới để tìm thông tin liên quan
- **BẮT BUỘC**: Khi sử dụng đường dẫn menu:
  + Nếu đường dẫn có chứa "mẫu mặc định" hoặc tên cụ thể của một tùy chọn (Mẫu 1, Mẫu 2...), hãy THAY THẾ bằng cách diễn đạt chung để khách hàng có thể chọn
  + VÍ DỤ SAI: "Mẫu hóa đơn > Mẫu hóa đơn 1: Khổ 80mm – mẫu mặc định"
  + VÍ DỤ ĐÚNG: "Mẫu hóa đơn > [Chọn mẫu hóa đơn bạn đang sử dụng]"
  + Chỉ giữ nguyên đường dẫn đến cấp cha (Menu chính > Submenu), không bao gồm tên cụ thể của tùy chọn
- Nếu Context có hướng dẫn từng bước, hãy trình bày rõ ràng với số thứ tự (Bước 1, Bước 2,...)
- **QUAN TRỌNG - PHÂN BIỆT MẪU VÀ CẤU HÌNH**:
  + **Mẫu/Tùy chọn cùng cấp** (Mẫu 1, 2, 3, 4...): KHÔNG nói "mặc định". Hãy liệt kê tất cả để khách hàng chọn.
    * VÍ DỤ: "Bước 1: Chọn mẫu hóa đơn bạn muốn (Mẫu 1, 2, 3, hoặc 4)"
  + **Cấu hình BÊN TRONG mẫu** (Cỡ chữ, Màu sắc...): CÓ THỂ nói "mặc định" và liệt kê tùy chọn.
    * VÍ DỤ: "Hệ thống mặc định cỡ chữ Nhỏ. Bạn có thể chọn Vừa hoặc Lớn."
- Nếu không tìm thấy thông tin trong Context, hãy nói rõ: "Tôi không tìm thấy thông tin về vấn đề này trong tài liệu hướng dẫn. Vui lòng liên hệ bộ phận hỗ trợ kỹ thuật để được giúp đỡ."

Định dạng câu trả lời mẫu cho hướng dẫn thao tác:

**VÍ DỤ 1 - Hướng dẫn chung cho nhiều mẫu:**
**Đường dẫn:** Cài đặt cửa hàng > Tối ưu bán hàng > Mẫu hóa đơn

**Các bước thực hiện:**
1. Truy cập: **Cài đặt cửa hàng > Tối ưu bán hàng > Mẫu hóa đơn**
2. Chọn mẫu hóa đơn bạn muốn cấu hình:
   - **Mẫu 1: Khổ 80mm – Mặc định**
   - **Mẫu 2: Khổ 80mm – Đường viền**
   - **Mẫu 3: Khổ 80mm – Đóng khung**
   - **Mẫu 4: Khổ A4/A5**
3. Sau khi chọn mẫu, cấu hình [Tính năng cụ thể]...

**VÍ DỤ 2 - Hướng dẫn cấu hình bên trong mẫu:**
**Đường dẫn:** Cài đặt cửa hàng > Tối ưu bán hàng > Mẫu hóa đơn > [Mẫu bạn đang dùng]

**Các bước thực hiện:**
1. Truy cập: **Cài đặt cửa hàng > Tối ưu bán hàng > Mẫu hóa đơn**
2. Chọn mẫu hóa đơn bạn đang sử dụng
3. Tại **[Tính năng]**, bạn có thể chọn:
   - **Nhỏ**: [Mô tả]
   - **Vừa**: [Mô tả] (khuyến nghị)
   - **Lớn**: [Mô tả]
   - **Lưu ý:** Hệ thống mặc định **Nhỏ**. Bạn nên chọn **Vừa** để dễ đọc hơn.
4. Nhấn **Lưu** để áp dụng.

**Lưu ý:** [Các lưu ý quan trọng]

Context (Tài liệu hướng dẫn):
{context}

Câu hỏi của khách hàng: {query}

Trả lời (bằng tiếng Việt, chi tiết, có cấu trúc rõ ràng, BẮT BUỘC bao gồm đường dẫn menu nếu Context có):"""

    try:
        response = ollama.generate(
            model=config.LLM_MODEL,
            prompt=prompt,
            options={
                "temperature": 0.7,
                "top_p": 0.9,
            }
        )
        return response['response']
    except Exception as e:
        print(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


def find_parent_chunks(chunk_metadata: Dict[str, Any], all_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find parent chunks in the hierarchy.
    Returns chunks that are ancestors of the current chunk.
    """
    parents = []
    current_heading_path = chunk_metadata.get("heading_path", "")
    current_level = chunk_metadata.get("level", 0)

    if not current_heading_path or current_level <= 1:
        return parents

    # Find chunks with lower level (higher in hierarchy) and matching path prefix
    for meta in all_metadata:
        if meta.get("level", 0) < current_level:
            parent_path = meta.get("heading_path", "")
            # Check if this is a parent (current path starts with parent path)
            if current_heading_path.startswith(parent_path) and parent_path:
                parents.append({
                    "heading": meta.get("heading", ""),
                    "heading_path": parent_path,
                    "content": meta.get("content", ""),
                    "level": meta.get("level", 0),
                    "type": "parent"
                })

    # Sort by level (closest parent first)
    parents.sort(key=lambda x: x["level"], reverse=True)

    return parents[:2]  # Return max 2 parent levels


def find_related_chunks(chunk_metadata: Dict[str, Any], all_metadata: List[Dict[str, Any]],
                       chunk_idx: int, max_related: int = 3) -> List[Dict[str, Any]]:
    """
    Find related chunks (siblings and children).
    IMPROVED: Uses siblings_raw and children_raw from Excel metadata when available.
    """
    related = []
    current_section_code = chunk_metadata.get("section_code", "")

    # Strategy 1: Use siblings_raw from Excel (PREFERRED - faster and more accurate)
    siblings_raw = chunk_metadata.get("siblings_raw", "")
    if siblings_raw and siblings_raw != "None":
        sibling_codes = [s.strip() for s in siblings_raw.split(",") if s.strip()]

        # Find chunks matching sibling section codes
        for meta in all_metadata:
            meta_section_code = meta.get("section_code", "")
            if meta_section_code in sibling_codes:
                related.append({
                    "heading": meta.get("heading", ""),
                    "heading_path": meta.get("heading_path", ""),
                    "content": meta.get("content", "")[:500] + "...",
                    "level": meta.get("level", 0),
                    "type": "sibling",
                    "section_code": meta_section_code,
                    "source": "excel_metadata"  # Mark as from Excel
                })

    # Strategy 2: Use children_raw from Excel (PREFERRED)
    children_raw = chunk_metadata.get("children_raw", "")
    if children_raw and children_raw != "None":
        children_codes = [c.strip() for c in children_raw.split(",") if c.strip()]

        # Find chunks matching children section codes
        for meta in all_metadata:
            meta_section_code = meta.get("section_code", "")
            if meta_section_code in children_codes:
                related.append({
                    "heading": meta.get("heading", ""),
                    "heading_path": meta.get("heading_path", ""),
                    "content": meta.get("content", "")[:500] + "...",
                    "level": meta.get("level", 0),
                    "type": "child",
                    "section_code": meta_section_code,
                    "source": "excel_metadata"
                })

    # Fallback: If no Excel metadata, use old logic (for backwards compatibility)
    if not related:
        current_heading_path = chunk_metadata.get("heading_path", "")
        current_level = chunk_metadata.get("level", 0)

        # Find siblings by heading path
        if " > " in current_heading_path:
            parent_path = " > ".join(current_heading_path.split(" > ")[:-1])

            for idx, meta in enumerate(all_metadata):
                if idx == chunk_idx:
                    continue

                meta_path = meta.get("heading_path", "")
                meta_level = meta.get("level", 0)

                # Same parent and same level = sibling
                if meta_level == current_level and " > " in meta_path:
                    meta_parent = " > ".join(meta_path.split(" > ")[:-1])
                    if meta_parent == parent_path:
                        related.append({
                            "heading": meta.get("heading", ""),
                            "heading_path": meta_path,
                            "content": meta.get("content", "")[:500] + "...",
                            "level": meta_level,
                            "type": "sibling",
                            "distance": abs(idx - chunk_idx),
                            "source": "calculated"
                        })

        # Find children by heading path
        for idx, meta in enumerate(all_metadata):
            if idx == chunk_idx:
                continue

            meta_path = meta.get("heading_path", "")
            meta_level = meta.get("level", 0)

            if meta_level == current_level + 1 and meta_path.startswith(current_heading_path):
                related.append({
                    "heading": meta.get("heading", ""),
                    "heading_path": meta_path,
                    "content": meta.get("content", "")[:500] + "...",
                    "level": meta_level,
                    "type": "child",
                    "distance": abs(idx - chunk_idx),
                    "source": "calculated"
                })

    # Sort by type priority (siblings first, then children)
    type_priority = {"sibling": 0, "child": 1}
    related.sort(key=lambda x: type_priority.get(x["type"], 2))

    return related[:max_related]


def detect_headings_and_hierarchy(text: str) -> List[Dict[str, Any]]:
    """
    Detect headings and create hierarchical chunks.
    Now with automatic size control - splits large sections into multiple chunks.
    """
    lines = text.split('\n')
    raw_sections = []
    current_heading = "Introduction"
    current_level = 0
    current_content = []
    heading_stack = [("Document", 0)]

    # Step 1: Parse document structure
    for line in lines:
        stripped = line.strip()

        if len(stripped) > 0 and len(stripped) < 100:
            is_heading = False
            level = 0

            # Detect headings (UPPERCASE or numbered)
            if stripped.isupper() and len(stripped.split()) <= 8:
                is_heading = True
                level = 1
            elif stripped[0].isdigit() and '.' in stripped[:5]:
                is_heading = True
                level = stripped.split()[0].count('.') + 1

            if is_heading:
                # Save previous section
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        raw_sections.append({
                            "heading": current_heading,
                            "level": current_level,
                            "heading_path": ' > '.join([h[0] for h in heading_stack]),
                            "content": content
                        })

                # Update heading stack
                while heading_stack and heading_stack[-1][1] >= level:
                    heading_stack.pop()

                current_heading = stripped
                current_level = level
                heading_stack.append((current_heading, level))
                current_content = []
                continue

        if stripped:
            current_content.append(line)

    # Save last section
    if current_content:
        content = '\n'.join(current_content).strip()
        if content:
            raw_sections.append({
                "heading": current_heading,
                "level": current_level,
                "heading_path": ' > '.join([h[0] for h in heading_stack]),
                "content": content
            })

    # Step 2: Split large sections into smaller chunks
    final_chunks = []
    for section in raw_sections:
        word_count = len(section["content"].split())

        # If section is too large, split it
        if word_count > config.MAX_CHUNK_SIZE:
            print(f"  📄 Section '{section['heading']}' has {word_count} words - splitting...")

            # Split the content
            sub_chunks = recursive_split_large_chunks(
                section["content"],
                heading=section["heading"]
            )

            # Create chunks with proper metadata
            for i, sub_chunk in enumerate(sub_chunks):
                final_chunks.append({
                    "heading": f"{section['heading']} (part {i+1}/{len(sub_chunks)})",
                    "level": section["level"],
                    "heading_path": section["heading_path"],
                    "content": sub_chunk,
                    "is_split": True,
                    "part_number": i + 1,
                    "total_parts": len(sub_chunks)
                })
            print(f"  ✓ Split into {len(sub_chunks)} chunks")

        else:
            # Section is fine as-is
            final_chunks.append({
                **section,
                "is_split": False
            })

    return final_chunks


@app.post("/api/documents/load-from-excel")
async def load_from_excel(
    excel_file_path: str = "data HDSD.xlsx",
    sheet_name: str = "Dữ liệu HDSD"
):
    """Load chunks directly from Excel file with rich metadata"""
    try:
        print(f"\n{'='*80}")
        print(f"📂 Loading chunks from Excel: {excel_file_path}")
        print(f"   Sheet: {sheet_name}")
        print(f"{'='*80}")

        # Check if pandas is available
        try:
            import pandas as pd
        except ImportError:
            raise HTTPException(status_code=500, detail="pandas not installed. Run: pip install pandas openpyxl")

        # Load Excel file
        excel_path = Path(__file__).parent / excel_file_path
        if not excel_path.exists():
            raise HTTPException(status_code=404, detail=f"Excel file not found: {excel_file_path}")

        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        print(f"\n✓ Loaded {len(df)} rows from Excel")

        chunks_added = 0
        chunk_sizes = []
        skipped = 0

        for idx, row in df.iterrows():
            # Extract content
            content = str(row.get('content', '')).strip()

            # Skip empty content
            if not content or content == 'nan':
                skipped += 1
                continue

            word_count = len(content.split())
            if word_count < config.MIN_CHUNK_SIZE:
                skipped += 1
                continue

            chunk_sizes.append(word_count)

            # Get embedding
            embedding = get_embedding(content)

            # Parse title path
            title_path = str(row.get('title_path', '')).strip()
            title_hierarchy = [t.strip() for t in title_path.split('>') if t.strip()] if title_path != 'nan' else []
            title = title_hierarchy[-1] if title_hierarchy else f"Chunk {idx+1}"

            # Calculate level
            section_code = str(row.get('section_code', '')).strip()
            level = section_code.count('.') + 1 if section_code != 'nan' else 0

            # Store with extended metadata
            vector_store["documents"].append(embedding)
            vector_store["metadata"].append({
                "filename": excel_file_path,
                "heading": title,
                "heading_path": title_path if title_path != 'nan' else "",
                "level": level,
                "content": content,
                "type": "excel_chunk",
                "word_count": word_count,
                # Extended metadata from Excel
                "chunk_id": idx + 1,
                "section_code": section_code if section_code != 'nan' else None,
                "chunk_index": int(row.get('chunk_index', 1)) if pd.notna(row.get('chunk_index')) else 1,
                "section_path": section_code if section_code != 'nan' else None,
                "parent_topic": str(row.get('parent_topic', '')).strip() if pd.notna(row.get('parent_topic')) else None,
                "siblings_raw": str(row.get('siblings_raw', '')).strip() if pd.notna(row.get('siblings_raw')) else None,
                "children_raw": str(row.get('children_raw', '')).strip() if pd.notna(row.get('children_raw')) else None,
                "business_module": str(row.get('business_module', '')).strip() if pd.notna(row.get('business_module')) else None,
                "tags_raw": str(row.get('tags_raw', '')).strip() if pd.notna(row.get('tags_raw')) else None,
                "related_entities_raw": str(row.get('related_entities_raw', '')).strip() if pd.notna(row.get('related_entities_raw')) else None,
                "chunk_type": "complete",
            })

            chunks_added += 1
            if chunks_added <= 10 or chunks_added % 10 == 0:
                print(f"  ✓ [{section_code:10s}] {title[:60]:60s} ({word_count:4d} words)")

        # Print statistics
        print(f"\n{'='*80}")
        print(f"📈 Loading Statistics:")
        print(f"  Rows in Excel: {len(df)}")
        print(f"  Chunks loaded: {chunks_added}")
        print(f"  Chunks skipped: {skipped}")
        if chunk_sizes:
            print(f"  Chunk sizes (words):")
            print(f"    - Min: {min(chunk_sizes)}")
            print(f"    - Max: {max(chunk_sizes)}")
            print(f"    - Average: {sum(chunk_sizes) / len(chunk_sizes):.1f}")
        print(f"  Total documents in store: {len(vector_store['documents'])}")
        print(f"{'='*80}\n")

        return {
            "message": f"Loaded chunks from Excel: {excel_file_path} (Sheet: {sheet_name})",
            "totalDocuments": len(vector_store["documents"]),
            "chunksAdded": chunks_added,
            "chunksSkipped": skipped,
            "chunkStats": {
                "min": min(chunk_sizes) if chunk_sizes else 0,
                "max": max(chunk_sizes) if chunk_sizes else 0,
                "avg": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                "total": chunks_added
            }
        }

    except Exception as e:
        print(f"\n❌ Error loading Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/load-from-json")
async def load_from_json(json_file_path: str = "tendoo_chunks_from_excel.json"):
    """Load pre-chunked documents from JSON file (supports both old and new format)"""
    try:
        print(f"\n{'='*80}")
        print(f"📂 Loading chunks from JSON: {json_file_path}")
        print(f"{'='*80}")

        # Load JSON file
        json_path = Path(__file__).parent / json_file_path
        if not json_path.exists():
            raise HTTPException(status_code=404, detail=f"JSON file not found: {json_file_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        if not isinstance(chunks_data, list):
            raise HTTPException(status_code=400, detail="JSON must be a list of chunks")

        chunks_added = 0
        chunk_sizes = []
        skipped = 0

        print(f"\n✓ Found {len(chunks_data)} chunks in JSON")

        for chunk_data in chunks_data:
            # Extract chunk information
            content = chunk_data.get("content", "")
            word_count = chunk_data.get("content_length", len(content.split()) if content else 0)

            # Skip empty chunks
            if not content or word_count < config.MIN_CHUNK_SIZE:
                skipped += 1
                continue

            chunk_sizes.append(word_count)

            # Get embedding
            embedding = get_embedding(content)

            # Build heading path from hierarchy
            hierarchy = chunk_data.get("title_hierarchy", [])
            heading_path = " > ".join(hierarchy) if hierarchy else chunk_data.get("hierarchy_text", "")

            # Store with metadata (support both old and new format)
            vector_store["documents"].append(embedding)
            vector_store["metadata"].append({
                "filename": "tendoo_document.docx",  # Source file
                "heading": chunk_data.get("title", ""),
                "heading_path": heading_path,
                "level": chunk_data.get("level", 0),
                "content": content,
                "type": "pre_chunked_json",
                "word_count": word_count,
                # Additional metadata from JSON (both old and new format)
                "chunk_id": chunk_data.get("chunk_id", 0),
                "section_path": chunk_data.get("section_path", ""),
                "section_code": chunk_data.get("section_code"),
                "chunk_type": chunk_data.get("chunk_type", "complete"),
                "paragraph_count": chunk_data.get("paragraph_count", 0),
                "chunk_index": chunk_data.get("chunk_index"),
                # Enhanced metadata (from new Excel-based format)
                "parent_topic": chunk_data.get("parent_topic"),
                "siblings_raw": chunk_data.get("siblings_raw"),
                "children_raw": chunk_data.get("children_raw"),
                "business_module": chunk_data.get("business_module"),
                "tags_raw": chunk_data.get("tags_raw"),
                "related_entities_raw": chunk_data.get("related_entities_raw"),
            })

            chunks_added += 1
            if chunks_added <= 10 or chunks_added % 10 == 0:
                print(f"  ✓ [{chunk_data.get('chunk_id', 0):3d}] {chunk_data.get('title', '')[:60]:60s} ({word_count:4d} words)")

        # Print statistics
        print(f"\n{'='*80}")
        print(f"📈 Loading Statistics:")
        print(f"  Chunks in JSON: {len(chunks_data)}")
        print(f"  Chunks loaded: {chunks_added}")
        print(f"  Chunks skipped: {skipped}")
        if chunk_sizes:
            print(f"  Chunk sizes (words):")
            print(f"    - Min: {min(chunk_sizes)}")
            print(f"    - Max: {max(chunk_sizes)}")
            print(f"    - Average: {sum(chunk_sizes) / len(chunk_sizes):.1f}")
        print(f"  Total documents in store: {len(vector_store['documents'])}")
        print(f"{'='*80}\n")

        return {
            "message": f"Loaded chunks from: {json_file_path}",
            "totalDocuments": len(vector_store["documents"]),
            "chunksAdded": chunks_added,
            "chunksSkipped": skipped,
            "chunkStats": {
                "min": min(chunk_sizes) if chunk_sizes else 0,
                "max": max(chunk_sizes) if chunk_sizes else 0,
                "avg": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                "total": chunks_added
            }
        }

    except Exception as e:
        print(f"\n❌ Error loading JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document with intelligent chunking"""
    try:
        print(f"\n{'='*80}")
        print(f"📁 Uploading: {file.filename}")
        print(f"{'='*80}")

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Check file extension
        file_ext = Path(file.filename).suffix.lower()

        # Use hierarchical chunking for DOCX files if enabled
        if file_ext == ".docx" and config.USE_HIERARCHICAL_CHUNKING:
            print(f"📄 Using hierarchical chunking for DOCX file")

            # Create hierarchical chunks directly from DOCX
            # Each section can be split into multiple chunks if content is too long
            hierarchical_chunks = hierarchical_chunker.create_chunks(
                tmp_path,
                max_chunk_size=config.MAX_CHUNK_SIZE
            )
            os.unlink(tmp_path)

            chunks_added = 0
            chunk_sizes = []

            print(f"\n✓ Processing {len(hierarchical_chunks)} hierarchical sections")

            for chunk_data in hierarchical_chunks:
                content = chunk_data["content"]
                word_count = chunk_data["word_count"]

                # Skip very small chunks
                if word_count < config.MIN_CHUNK_SIZE:
                    print(f"  ⊘ Skipping tiny chunk: '{chunk_data['title']}' ({word_count} words)")
                    continue

                chunk_sizes.append(word_count)

                # Get embedding
                embedding = get_embedding(content)

                # Store with full metadata
                vector_store["documents"].append(embedding)
                vector_store["metadata"].append({
                    "filename": file.filename,
                    "heading": chunk_data["title"],
                    "heading_path": chunk_data["title_path"],
                    "level": chunk_data["level"],
                    "content": content,
                    "type": "hierarchical",
                    "word_count": word_count,
                    # Extended metadata for hierarchical chunks
                    "section_code": chunk_data["section_code"],
                    "chunk_index": chunk_data["chunk_index"],
                    "parent_topic": chunk_data["parent_topic"],
                    "siblings_raw": chunk_data["siblings_raw"],
                    "children_raw": chunk_data["children_raw"],
                    "business_module": chunk_data["business_module"],
                    "tags_raw": chunk_data["tags_raw"],
                    "related_entities_raw": chunk_data["related_entities_raw"],
                    "is_split": chunk_data.get("is_split", False),
                    "total_parts": chunk_data.get("total_parts", 1),
                })

                chunks_added += 1
                print(f"  ✓ [{chunk_data['section_code']:10s}] {chunk_data['title'][:50]:50s} | {chunk_data['business_module']:12s}")

            # Print statistics
            print(f"\n{'='*80}")
            print(f"📈 Chunking Statistics:")
            print(f"  Total chunks created: {chunks_added}")
            if chunk_sizes:
                print(f"  Chunk sizes (words):")
                print(f"    - Min: {min(chunk_sizes)}")
                print(f"    - Max: {max(chunk_sizes)}")
                print(f"    - Average: {sum(chunk_sizes) / len(chunk_sizes):.1f}")
            print(f"  Total documents in store: {len(vector_store['documents'])}")
            print(f"{'='*80}\n")

            return {
                "message": f"Uploaded and indexed: {file.filename}",
                "totalDocuments": len(vector_store["documents"]),
                "chunksAdded": chunks_added,
                "chunkStats": {
                    "min": min(chunk_sizes) if chunk_sizes else 0,
                    "max": max(chunk_sizes) if chunk_sizes else 0,
                    "avg": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                    "total": chunks_added
                }
            }

        # Extract text for non-DOCX files or if hierarchical chunking is disabled
        text = extract_text_from_file(tmp_path, file.filename)
        os.unlink(tmp_path)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty document")

        total_words = len(text.split())
        print(f"📊 Document stats: {total_words} words")

        # Try hierarchical chunking first
        hierarchical_chunks = detect_headings_and_hierarchy(text)

        chunks_added = 0
        chunk_sizes = []

        if len(hierarchical_chunks) > 0:
            print(f"\n✓ Detected {len(hierarchical_chunks)} hierarchical sections")

            for chunk_data in hierarchical_chunks:
                content = chunk_data["content"]
                word_count = len(content.split())

                # Skip very small chunks
                if word_count < config.MIN_CHUNK_SIZE:
                    print(f"  ⊘ Skipping tiny chunk: '{chunk_data['heading']}' ({word_count} words)")
                    continue

                chunk_sizes.append(word_count)

                # Get embedding
                embedding = get_embedding(content)

                # Store
                vector_store["documents"].append(embedding)
                vector_store["metadata"].append({
                    "filename": file.filename,
                    "heading": chunk_data["heading"],
                    "heading_path": chunk_data["heading_path"],
                    "level": chunk_data["level"],
                    "content": content,
                    "type": "hierarchical",
                    "word_count": word_count,
                    "is_split": chunk_data.get("is_split", False)
                })

                chunks_added += 1
                print(f"  ✓ Added: '{chunk_data['heading']}' ({word_count} words)")

        else:
            # Fallback to sentence-based chunking
            print(f"\n⚠ No headings detected - using sentence-based chunking")
            chunks = chunk_text(text, config.TARGET_CHUNK_SIZE, config.CHUNK_OVERLAP)

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                word_count = len(chunk.split())
                chunk_sizes.append(word_count)

                embedding = get_embedding(chunk)

                vector_store["documents"].append(embedding)
                vector_store["metadata"].append({
                    "filename": file.filename,
                    "heading": f"Chunk {i+1}",
                    "heading_path": f"{file.filename} > Chunk {i+1}",
                    "level": 0,
                    "content": chunk,
                    "type": "simple",
                    "word_count": word_count
                })

                chunks_added += 1

        # Print statistics
        print(f"\n{'='*80}")
        print(f"📈 Chunking Statistics:")
        print(f"  Total chunks created: {chunks_added}")
        if chunk_sizes:
            print(f"  Chunk sizes (words):")
            print(f"    - Min: {min(chunk_sizes)}")
            print(f"    - Max: {max(chunk_sizes)}")
            print(f"    - Average: {sum(chunk_sizes) / len(chunk_sizes):.1f}")
            print(f"    - Target range: {config.MIN_CHUNK_SIZE}-{config.MAX_CHUNK_SIZE}")
        print(f"  Total documents in store: {len(vector_store['documents'])}")
        print(f"{'='*80}\n")

        return {
            "message": f"Uploaded and indexed: {file.filename}",
            "totalDocuments": len(vector_store["documents"]),
            "chunksAdded": chunks_added,
            "chunkStats": {
                "min": min(chunk_sizes) if chunk_sizes else 0,
                "max": max(chunk_sizes) if chunk_sizes else 0,
                "avg": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                "total": chunks_added
            }
        }

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using Enhanced RAG with Hybrid Search + Reranking"""
    try:
        if not vector_store["documents"]:
            raise HTTPException(status_code=400, detail="No documents indexed yet")

        print(f"\n{'='*80}")
        print(f"Query: {request.query}")
        print(f"Hybrid Search: {config.USE_HYBRID_SEARCH}")
        print(f"Reranking: {config.USE_RERANKING}")
        print(f"Deduplication: {config.USE_DEDUPLICATION}")
        print(f"Context Enrichment: {config.USE_CONTEXT_ENRICHMENT}")
        print(f"{'='*80}\n")

        # Get query embedding
        query_embedding = get_embedding(request.query)

        # Stage 1: Retrieval
        retrieve_count = request.topK * config.RERANKING_MULTIPLIER if config.USE_RERANKING else request.topK

        doc_embeddings = np.array(vector_store["documents"])
        dense_scores = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)[0]

        if config.USE_HYBRID_SEARCH:
            # Sparse retrieval (BM25)
            documents_text = [meta['content'] for meta in vector_store["metadata"]]
            sparse_scores = BM25.calculate_bm25_scores(request.query, documents_text,
                                                       config.BM25_K1, config.BM25_B)

            # Reciprocal Rank Fusion
            fused_scores = reciprocal_rank_fusion(dense_scores.tolist(), sparse_scores, config.RRF_K)
            top_indices = np.argsort(fused_scores)[-retrieve_count:][::-1]

            print(f"Stage 1 - Hybrid retrieval: {len(top_indices)} candidates")
        else:
            top_indices = np.argsort(dense_scores)[-retrieve_count:][::-1]
            print(f"Stage 1 - Dense retrieval: {len(top_indices)} candidates")

        # Prepare candidate documents
        candidate_docs = []
        for idx in top_indices:
            metadata = vector_store["metadata"][idx]
            candidate_docs.append({
                "idx": idx,
                "content": metadata["content"],
                "heading": metadata["heading"],
                "heading_path": metadata["heading_path"],
                "level": metadata.get("level", 0),
                "filename": metadata["filename"],
                "similarity": float(dense_scores[idx])
            })

        # Stage 2: Deduplication
        if config.USE_DEDUPLICATION:
            candidate_docs = deduplicate_documents(candidate_docs, config.DEDUP_THRESHOLD)
            print(f"Stage 2 - After deduplication: {len(candidate_docs)} unique chunks")
        else:
            print(f"Stage 2 - Skipped deduplication")

        # Stage 3: Reranking
        if config.USE_RERANKING and len(candidate_docs) > request.topK:
            final_docs = rerank_documents(request.query, candidate_docs, request.topK)
            print(f"Stage 3 - After reranking: {len(final_docs)} top documents")
        else:
            final_docs = candidate_docs[:request.topK]
            print(f"Stage 3 - Skipped reranking")

        # Build context structure with parent and related chunks
        context_chunks = []
        context_metadata = []  # Track metadata for each context chunk
        context_structure_chunks = []
        retrieved_docs = []
        parent_chunks_list = []
        related_chunks_list = []

        if config.USE_CONTEXT_ENRICHMENT:
            print(f"\nEnriching context with parent and related chunks...")

        for doc in final_docs:
            doc_idx = doc['idx']
            doc_metadata = vector_store["metadata"][doc_idx]

            # Add primary chunk with its metadata
            context_chunks.append(doc['content'])
            context_metadata.append(doc_metadata)  # Store metadata for LLM context

            primary_chunk = {
                "type": "primary",
                "heading": doc['heading'],
                "headingPath": doc['heading_path'],
                "content": doc['content'],
                "similarity": doc['similarity'],
                "importance": 0.8,
                "contextType": "match"  # This is a direct match
            }
            context_structure_chunks.append(primary_chunk)

            # Find parent and related chunks (if enrichment is enabled)
            if config.USE_CONTEXT_ENRICHMENT:
                # Find parent chunks
                parents = find_parent_chunks(doc_metadata, vector_store["metadata"])
                if parents:
                    print(f"  📌 Found {len(parents)} parent(s) for '{doc['heading']}'")
                    for parent in parents:
                        parent_chunk = {
                            "type": "primary",  # Frontend compatibility
                            "contextType": "parent",  # Backend tracking
                            "heading": parent["heading"],
                            "headingPath": parent["heading_path"],
                            "content": parent["content"][:800] + "...",  # Truncate for brevity
                            "level": parent["level"],
                            "relatedTo": doc['heading']
                        }
                        parent_chunks_list.append(parent_chunk)

                # Find related chunks
                related = find_related_chunks(doc_metadata, vector_store["metadata"], doc_idx, max_related=2)
                if related:
                    print(f"  🔗 Found {len(related)} related chunk(s) for '{doc['heading']}'")
                    for rel in related:
                        related_chunk = {
                            "type": "primary",  # Frontend compatibility
                            "contextType": "related",  # Backend tracking
                            "heading": rel["heading"],
                            "headingPath": rel["heading_path"],
                            "content": rel["content"],
                            "level": rel["level"],
                            "relationshipType": rel["type"],  # sibling or child
                            "relatedTo": doc['heading']
                        }
                        related_chunks_list.append(related_chunk)

            retrieved_docs.append({
                "filename": doc['filename'],
                "content": doc['content'],
                "similarity": doc['similarity']
            })

        # Add unique parent and related chunks to context structure
        context_structure_chunks.extend(parent_chunks_list)
        context_structure_chunks.extend(related_chunks_list)

        if config.USE_CONTEXT_ENRICHMENT:
            print(f"\n📊 Context Summary:")
            print(f"  Primary chunks: {len(final_docs)}")
            print(f"  Parent chunks: {len(parent_chunks_list)}")
            print(f"  Related chunks: {len(related_chunks_list)}")
            print(f"  Total context: {len(context_structure_chunks)} chunks\n")

        # Generate answer with rich metadata
        answer = generate_answer(request.query, context_chunks, context_metadata)

        return QueryResponse(
            answer=answer,
            retrievedDocuments=retrieved_docs,
            contextStructure={
                "chunks": context_structure_chunks,
                "summary": {
                    "primaryCount": len(final_docs),
                    "parentCount": len(parent_chunks_list),
                    "relatedCount": len(related_chunks_list),
                    "totalCount": len(context_structure_chunks)
                }
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/count")
async def get_document_count():
    """Get total number of indexed documents"""
    return {"count": len(vector_store["documents"])}


@app.get("/api/documents/chunks")
async def get_all_chunks(
    filename: Optional[str] = None,
    include_content: bool = False,
    limit: Optional[int] = None,
    offset: int = 0
):
    """
    Get all chunks with detailed metadata

    Parameters:
    - filename: Filter by filename (optional)
    - include_content: Include full content (default: False, shows preview only)
    - limit: Maximum number of chunks to return (optional)
    - offset: Number of chunks to skip (default: 0)
    """
    if not vector_store["metadata"]:
        return {
            "total": 0,
            "chunks": [],
            "summary": {}
        }

    # Filter chunks
    all_chunks = []
    for idx, metadata in enumerate(vector_store["metadata"]):
        # Apply filename filter if specified
        if filename and metadata.get("filename") != filename:
            continue

        chunk_info = {
            "index": idx,
            "filename": metadata.get("filename", "Unknown"),
            "heading": metadata.get("heading", ""),
            "headingPath": metadata.get("heading_path", ""),
            "level": metadata.get("level", 0),
            "type": metadata.get("type", "unknown"),
            "wordCount": metadata.get("word_count", 0),
            "isSplit": metadata.get("is_split", False),
            # Hierarchical metadata
            "section_code": metadata.get("section_code", ""),
            "chunk_index": metadata.get("chunk_index", 1),
            "parent_topic": metadata.get("parent_topic", ""),
            "siblings_raw": metadata.get("siblings_raw", ""),
            "children_raw": metadata.get("children_raw", ""),
            "business_module": metadata.get("business_module", ""),
            "tags_raw": metadata.get("tags_raw", ""),
            "related_entities_raw": metadata.get("related_entities_raw", ""),
            "total_parts": metadata.get("total_parts", 1),
        }

        # Include content based on parameter
        if include_content:
            chunk_info["content"] = metadata.get("content", "")
        else:
            # Show preview only (first 200 chars)
            content = metadata.get("content", "")
            chunk_info["contentPreview"] = content[:200] + "..." if len(content) > 200 else content
            chunk_info["contentLength"] = len(content)

        # Add split info if applicable
        if metadata.get("is_split"):
            chunk_info["partNumber"] = metadata.get("part_number", 0)
            chunk_info["totalParts"] = metadata.get("total_parts", 0)

        all_chunks.append(chunk_info)

    # Apply pagination
    total_chunks = len(all_chunks)
    start_idx = offset
    end_idx = offset + limit if limit else len(all_chunks)
    paginated_chunks = all_chunks[start_idx:end_idx]

    # Calculate summary statistics
    filenames = set(chunk["filename"] for chunk in all_chunks)
    word_counts = [chunk["wordCount"] for chunk in all_chunks if chunk["wordCount"] > 0]

    summary = {
        "totalChunks": total_chunks,
        "uniqueFiles": len(filenames),
        "files": list(filenames),
        "chunkSizeStats": {
            "min": min(word_counts) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
            "avg": sum(word_counts) / len(word_counts) if word_counts else 0,
        },
        "typeDistribution": {},
        "levelDistribution": {}
    }

    # Count types and levels
    for chunk in all_chunks:
        chunk_type = chunk.get("type", "unknown")
        chunk_level = chunk.get("level", 0)

        summary["typeDistribution"][chunk_type] = summary["typeDistribution"].get(chunk_type, 0) + 1
        summary["levelDistribution"][str(chunk_level)] = summary["levelDistribution"].get(str(chunk_level), 0) + 1

    return {
        "total": total_chunks,
        "showing": len(paginated_chunks),
        "offset": offset,
        "limit": limit,
        "chunks": paginated_chunks,
        "summary": summary
    }


@app.delete("/api/documents")
async def clear_documents():
    """Clear all documents"""
    vector_store["documents"].clear()
    vector_store["metadata"].clear()
    return {"message": "All documents cleared successfully"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "RAG Demo API - Enhanced with Context Enrichment",
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "gemini_model": config.GEMINI_MODEL if config.GEMINI_API_KEY else "Not configured",
        "features": {
            "hybrid_search": config.USE_HYBRID_SEARCH,
            "reranking": config.USE_RERANKING,
            "deduplication": config.USE_DEDUPLICATION,
            "semantic_chunking": config.USE_SEMANTIC_CHUNKING,
            "context_enrichment": config.USE_CONTEXT_ENRICHMENT
        },
        "chunking_config": {
            "min_chunk_size": config.MIN_CHUNK_SIZE,
            "target_chunk_size": config.TARGET_CHUNK_SIZE,
            "max_chunk_size": config.MAX_CHUNK_SIZE,
            "overlap": config.CHUNK_OVERLAP
        },
        "documents": len(vector_store["documents"])
    }


@app.get("/chunks-viewer", response_class=HTMLResponse)
async def chunks_viewer():
    """Serve the chunks viewer HTML page"""
    html_path = Path(__file__).parent / "chunks_viewer.html"

    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Chunks viewer page not found")

    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
