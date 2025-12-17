# -*- coding: utf-8 -*-
"""
Tendoo Customer Support Chatbot - RAG System
Chatbot hỗ trợ khách hàng về Tendoo App
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import google.generativeai as genai
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import re
import math
from collections import Counter, defaultdict
from datetime import datetime

app = FastAPI(title="Tendoo Customer Support Chatbot")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration
class Config:
    # API Keys
    GEMINI_API_KEY = "AIzaSyCkNVdwWEsJGmwc3NyrKXAnv62i45LsSWw"
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "bge-m3"

    # File paths
    CHUNKS_FILE = "output_tendoo/chunks.json"

    # Feature flags
    USE_HYBRID_SEARCH = True  # Dense + BM25
    USE_GRAPH_ENRICHMENT = True  # Include parent/children/siblings
    USE_RERANKING = True  # LLM reranking
    USE_QUERY_EXPANSION = False  # Keep it simple

    # RAG parameters
    TOP_K = 5  # Top chunks to retrieve
    MAX_DESCENDANTS = 5  # Max children to include
    MAX_SIBLINGS = 3  # Max siblings to include
    INCLUDE_PARENT = True

    # Smart selection parameters
    MIN_DESCENDANT_SCORE = 0.3  # Minimum score to keep descendant
    MIN_SIBLING_SCORE = 0.4  # Minimum score to keep sibling

    # Enriched embeddings
    USE_ENRICHED_EMBEDDINGS = True
    PARENT_CONTEXT_LENGTH = 200  # Max chars from parent

    # BM25 parameters
    BM25_K1 = 1.5
    BM25_B = 0.75
    RRF_K = 60  # For hybrid fusion

    # Reranking
    RERANK_TOP_K = 3  # Final number after reranking


config = Config()

# Configure Gemini API
genai.configure(api_key=config.GEMINI_API_KEY)


# In-memory storage
vector_store = {
    "chunks": [],
    "embeddings": [],
    "chunk_map": {},
    "chat_history": []  # Store recent conversations
}


# Pydantic Models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    include_history: bool = False


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None


# ==================== BM25 IMPLEMENTATION ====================

class BM25:
    """BM25 sparse retrieval"""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize Vietnamese text"""
        if not text:
            return []
        # Remove punctuation, lowercase
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [word for word in cleaned.split() if len(word) > 1]
        return tokens

    @staticmethod
    def calculate_bm25_scores(query: str, documents: List[str],
                             k1: float = 1.5, b: float = 0.75) -> List[float]:
        """Calculate BM25 scores"""
        query_terms = BM25.tokenize(query)
        if not query_terms:
            return [0.0] * len(documents)

        # Tokenize all documents
        doc_tokens = [BM25.tokenize(doc) for doc in documents]

        # Calculate average document length
        doc_lengths = [len(tokens) for tokens in doc_tokens]
        avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

        # Calculate IDF
        idf_scores = {}
        total_docs = len(documents)
        for term in query_terms:
            docs_with_term = sum(1 for tokens in doc_tokens if term in tokens)
            idf = math.log((total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1.0)
            idf_scores[term] = idf

        # Calculate BM25 scores
        scores = []
        for doc_idx, tokens in enumerate(doc_tokens):
            score = 0.0
            doc_len = doc_lengths[doc_idx]
            term_freqs = Counter(tokens)

            for term in query_terms:
                if term not in term_freqs:
                    continue

                tf = term_freqs[term]
                idf = idf_scores[term]

                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                score += idf * (numerator / denominator)

            scores.append(score)

        return scores


# ==================== HELPER FUNCTIONS ====================

def load_chunks():
    """Load chunks từ file JSON"""
    chunks_file = config.CHUNKS_FILE

    if not os.path.exists(chunks_file):
        print(f"❌ File {chunks_file} không tồn tại!")
        print("Hãy chạy test_tendoo.py trước để tạo chunks.")
        return []

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"✅ Đã load {len(chunks)} chunks từ {chunks_file}")
    return chunks


def create_embedding(text: str) -> List[float]:
    """Tạo embedding sử dụng Ollama BGE-M3"""
    try:
        response = ollama.embeddings(
            model=config.EMBEDDING_MODEL,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        print(f"❌ Lỗi tạo embedding: {e}")
        return []


def build_enriched_text_for_embedding(chunk_data: Dict) -> str:
    """
    Build enriched text including parent context for embedding generation.
    This ensures hierarchical context is included in the embedding space.

    Example output:
        "[NGỮ CẢNH: Quản lý sản phẩm]
         [VỊ TRÍ: Hướng dẫn sử dụng > Quản lý sản phẩm > Thêm sản phẩm mới]
         Để thêm sản phẩm mới, bạn vào mục Sản phẩm..."

    Args:
        chunk_data: Chunk dictionary with 'content' and 'metadata'

    Returns:
        Enriched text combining parent context + title path + main content
    """
    if not config.USE_ENRICHED_EMBEDDINGS:
        return chunk_data.get('content', '').strip()

    metadata = chunk_data.get('metadata', {})
    content = chunk_data.get('content', '').strip()
    enriched_parts = []

    # 1. Add parent context (if exists)
    parent_id = metadata.get('parent_id')
    if parent_id and parent_id in vector_store['chunk_map']:
        parent_chunk = vector_store['chunk_map'][parent_id]
        parent_content = parent_chunk.get('content', '')[:config.PARENT_CONTEXT_LENGTH]
        parent_title = parent_chunk.get('metadata', {}).get('section_title', '')

        if parent_title:
            enriched_parts.append(f"[NGỮ CẢNH: {parent_title}]")
        if parent_content:
            enriched_parts.append(parent_content)

    # 2. Add title path (hierarchical breadcrumb)
    title_path = metadata.get('title_path', [])
    if title_path and len(title_path) > 1:
        path_str = ' > '.join(title_path[-3:])  # Last 3 levels
        enriched_parts.append(f"[VỊ TRÍ: {path_str}]")

    # 3. Add main content
    enriched_parts.append(content)

    return ' '.join(enriched_parts)


def embed_chunks(chunks: List[Dict]):
    """Tạo embeddings cho tất cả chunks với enriched text"""
    print("\n🔄 Đang tạo embeddings cho chunks...")

    embeddings = []
    for i, chunk in enumerate(chunks):
        # Build enriched text with parent context
        enriched_text = build_enriched_text_for_embedding(chunk)
        embedding = create_embedding(enriched_text)

        if embedding:
            embeddings.append(embedding)
        else:
            embeddings.append([0.0] * 1024)

        if (i + 1) % 10 == 0:
            print(f"  Đã embed {i + 1}/{len(chunks)} chunks")

    print(f"✅ Đã tạo {len(embeddings)} embeddings (với enriched context)")
    return embeddings


def hybrid_search(query: str, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
    """Hybrid search: Dense (embedding) + Sparse (BM25) với RRF fusion"""
    chunks = vector_store["chunks"]
    embeddings = vector_store["embeddings"]

    if not chunks or not embeddings:
        return []

    # 1. Dense retrieval (cosine similarity)
    query_emb = np.array(query_embedding).reshape(1, -1)
    chunk_embs = np.array(embeddings)
    dense_scores = cosine_similarity(query_emb, chunk_embs)[0]

    # 2. Sparse retrieval (BM25)
    documents = [f"{c['metadata']['section_title']} {c['content']}" for c in chunks]
    sparse_scores = BM25.calculate_bm25_scores(query, documents,
                                               k1=config.BM25_K1,
                                               b=config.BM25_B)

    # 3. RRF Fusion
    dense_ranks = np.argsort(dense_scores)[::-1]
    sparse_ranks = np.argsort(sparse_scores)[::-1]

    rrf_scores = defaultdict(float)
    k = config.RRF_K

    for rank, idx in enumerate(dense_ranks):
        rrf_scores[idx] += 1.0 / (k + rank + 1)

    for rank, idx in enumerate(sparse_ranks):
        rrf_scores[idx] += 1.0 / (k + rank + 1)

    # Sort by RRF score
    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Get top K
    results = []
    for idx in sorted_indices[:top_k]:
        results.append({
            "chunk": chunks[idx],
            "score": float(rrf_scores[idx]),
            "dense_score": float(dense_scores[idx]),
            "sparse_score": float(sparse_scores[idx])
        })

    return results


def score_descendant_relevance(descendant: Dict, query: str, query_embedding: List[float]) -> float:
    """
    Tính điểm liên quan của descendant với query
    Kết hợp: semantic similarity + keyword matching
    """
    # Semantic similarity
    desc_embedding = create_embedding(descendant['content'])
    if not desc_embedding or not query_embedding:
        return 0.0

    query_emb = np.array(query_embedding).reshape(1, -1)
    desc_emb = np.array(desc_embedding).reshape(1, -1)
    semantic_score = cosine_similarity(query_emb, desc_emb)[0][0]

    # Keyword matching (BM25-like)
    query_terms = BM25.tokenize(query)
    desc_terms = BM25.tokenize(descendant['content'])

    if not query_terms or not desc_terms:
        keyword_score = 0.0
    else:
        matched = len(set(query_terms) & set(desc_terms))
        keyword_score = matched / len(query_terms)

    # Combine (70% semantic, 30% keyword)
    final_score = 0.7 * semantic_score + 0.3 * keyword_score

    return final_score


def find_smart_descendants(
    chunk: Dict,
    query: str,
    query_embedding: List[float],
    max_descendants: int = 5,
    min_score: float = 0.3
) -> List[tuple]:
    """
    Tìm descendants có score cao nhất (thay vì lấy tất cả)
    Trả về: List of (descendant, score)
    """
    chunk_map = vector_store["chunk_map"]
    all_descendants = []
    visited = set()

    def collect_descendants(current_chunk: Dict):
        children_ids = current_chunk.get('metadata', {}).get('children_ids', [])

        for child_id in children_ids:
            if child_id in visited or child_id not in chunk_map:
                continue

            visited.add(child_id)
            child_chunk = chunk_map[child_id]
            all_descendants.append(child_chunk)

            # Đệ quy
            collect_descendants(child_chunk)

    collect_descendants(chunk)

    if not all_descendants:
        return []

    # Score tất cả descendants
    scored_descendants = []
    for desc in all_descendants:
        score = score_descendant_relevance(desc, query, query_embedding)
        if score >= min_score:
            scored_descendants.append((desc, score))

    # Sort by score và lấy top K
    scored_descendants.sort(key=lambda x: x[1], reverse=True)
    return scored_descendants[:max_descendants]


def find_smart_siblings(
    chunk: Dict,
    query: str,
    query_embedding: List[float],
    max_siblings: int = 3,
    min_score: float = 0.4
) -> List[tuple]:
    """
    Tìm siblings (anh chị em cùng cấp) có score cao nhất
    Trả về: List of (sibling, score)
    """
    chunk_map = vector_store["chunk_map"]
    sibling_ids = chunk.get('metadata', {}).get('sibling_ids', [])

    if not sibling_ids:
        return []

    # Score tất cả siblings
    scored_siblings = []
    for sibling_id in sibling_ids:
        if sibling_id not in chunk_map:
            continue

        sibling_chunk = chunk_map[sibling_id]
        score = score_descendant_relevance(sibling_chunk, query, query_embedding)

        if score >= min_score:
            scored_siblings.append((sibling_chunk, score))

    # Sort by score và lấy top K
    scored_siblings.sort(key=lambda x: x[1], reverse=True)
    return scored_siblings[:max_siblings]


def enrich_with_graph(retrieved_chunks: List[Dict], query: str, query_embedding: List[float]) -> List[Dict]:
    """Làm giàu context với parent, children (smart), siblings (smart)"""
    chunk_map = vector_store["chunk_map"]
    enriched = []

    for item in retrieved_chunks:
        chunk = item["chunk"]
        score = item["score"]

        # Main chunk
        enriched.append({
            "type": "main",
            "chunk_id": chunk["chunk_id"],
            "section_code": chunk["metadata"]["section_code"],
            "section_title": chunk["metadata"]["section_title"],
            "content": chunk["content"],
            "score": score,
            "title_path": " > ".join(chunk["metadata"]["title_path"])
        })

        # Parent (if enabled)
        if config.INCLUDE_PARENT and chunk["metadata"]["parent_id"]:
            parent_id = chunk["metadata"]["parent_id"]
            if parent_id in chunk_map:
                parent = chunk_map[parent_id]
                enriched.append({
                    "type": "parent",
                    "chunk_id": parent["chunk_id"],
                    "section_code": parent["metadata"]["section_code"],
                    "section_title": parent["metadata"]["section_title"],
                    "content": parent["content"][:200] + "...",
                    "title_path": " > ".join(parent["metadata"]["title_path"])
                })

        # Smart descendants (chỉ lấy những cái liên quan)
        smart_descendants = find_smart_descendants(
            chunk, query, query_embedding,
            max_descendants=config.MAX_DESCENDANTS,
            min_score=config.MIN_DESCENDANT_SCORE
        )

        for desc, desc_score in smart_descendants:
            enriched.append({
                "type": "child",
                "chunk_id": desc["chunk_id"],
                "section_code": desc["metadata"]["section_code"],
                "section_title": desc["metadata"]["section_title"],
                "content": desc["content"],
                "title_path": " > ".join(desc["metadata"]["title_path"]),
                "relevance_score": desc_score
            })

        # Smart siblings (chỉ lấy những cái liên quan)
        smart_siblings = find_smart_siblings(
            chunk, query, query_embedding,
            max_siblings=config.MAX_SIBLINGS,
            min_score=config.MIN_SIBLING_SCORE
        )

        for sib, sib_score in smart_siblings:
            enriched.append({
                "type": "sibling",
                "chunk_id": sib["chunk_id"],
                "section_code": sib["metadata"]["section_code"],
                "section_title": sib["metadata"]["section_title"],
                "content": sib["content"][:150] + "...",
                "title_path": " > ".join(sib["metadata"]["title_path"]),
                "relevance_score": sib_score
            })

    print(f"   🔗 Enriched: {len(enriched)} chunks (main + parents + smart descendants + smart siblings)")
    return enriched


def rerank_with_llm(query: str, chunks: List[Dict]) -> List[Dict]:
    """Rerank chunks using LLM"""
    if len(chunks) <= config.RERANK_TOP_K:
        return chunks

    # Create prompt for reranking
    chunks_text = []
    for i, chunk in enumerate(chunks[:15]):  # Only rerank top 15
        chunks_text.append(
            f"[{i}] {chunk['section_code']}: {chunk['section_title']}\n{chunk['content'][:200]}"
        )

    prompt = f"""Bạn là AI chuyên đánh giá mức độ liên quan của tài liệu.

Câu hỏi: "{query}"

Các đoạn văn:
{chr(10).join(chunks_text)}

Hãy chọn {config.RERANK_TOP_K} đoạn văn LIÊN QUAN NHẤT để trả lời câu hỏi.
Trả về danh sách chỉ số (số nguyên) cách nhau bởi dấu phẩy.
Ví dụ: 0,3,7

Chỉ trả về số, không giải thích."""

    try:
        model = genai.GenerativeModel(config.LLM_MODEL)
        print(prompt)
        response = model.generate_content(prompt)

        # Parse indices
        indices_str = response.text.strip()
        indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]

        # Return reranked chunks
        reranked = [chunks[i] for i in indices if i < len(chunks)]
        return reranked[:config.RERANK_TOP_K]

    except Exception as e:
        print(f"⚠️ Lỗi reranking: {e}")
        return chunks[:config.RERANK_TOP_K]


def generate_answer(query: str, context_chunks: List[Dict]) -> str:
    """Sinh câu trả lời sử dụng Gemini"""

    # Tạo context string
    context_parts = []
    for chunk in context_chunks:
        context_parts.append(
            f"📍 [{chunk.get('type', 'main').upper()}] {chunk['section_code']}: {chunk['section_title']}\n"
            f"Đường dẫn: {chunk['title_path']}\n"
            f"Nội dung: {chunk['content']}\n"
        )

    context_str = "\n" + "="*80 + "\n".join(context_parts)

    # Prompt cho chatbot chăm sóc khách hàng Tendoo
    prompt = f"""Bạn là trợ lý AI chăm sóc khách hàng của Tendoo App - Ứng dụng quản lý cửa hàng và bán hàng.

NHIỆM VỤ CỦA BẠN:
- Hỗ trợ khách hàng sử dụng Tendoo App
- Trả lời các câu hỏi về tính năng, cách sử dụng, cài đặt
- Hướng dẫn chi tiết, từng bước một
- Thân thiện, nhiệt tình, chuyên nghiệp

THÔNG TIN TÀI LIỆU:
{context_str}

CÂU HỎI KHÁCH HÀNG: {query}

HƯỚNG DẪN TRẢ LỜI:
1. Chào hỏi thân thiện (nếu cần)
2. Trả lời CHÍNH XÁC dựa trên tài liệu
3. Nếu có các bước hướng dẫn:
   - Liệt kê rõ ràng từng bước
   - Sử dụng số thứ tự hoặc dấu đầu dòng
4. Nếu có thông tin quan trọng, nhấn mạnh bằng "⚠️ Lưu ý:"
5. Kết thúc bằng câu hỏi "Bạn còn thắc mắc gì khác không?" (nếu phù hợp)
6. Nếu KHÔNG tìm thấy thông tin trong tài liệu:
   - Nói rõ "Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu"
   - Đề xuất liên hệ bộ phận hỗ trợ

CHÚ Ý:
- KHÔNG bịa đặt thông tin không có trong tài liệu
- Sử dụng tiếng Việt tự nhiên, dễ hiểu
- Tránh thuật ngữ kỹ thuật khó hiểu

TRẢ LỜI:"""

    try:
        model = genai.GenerativeModel(config.LLM_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Lỗi khi sinh câu trả lời: {e}")
        return f"Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn. Vui lòng thử lại sau.\n\nLỗi: {str(e)}"


# ==================== API ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    """Load chunks và tạo embeddings khi khởi động"""
    print("\n" + "="*80)
    print("KHỞI ĐỘNG TENDOO CUSTOMER SUPPORT CHATBOT")
    print("="*80)

    # Load chunks
    chunks = load_chunks()
    if not chunks:
        print("⚠️ Không có chunks nào được load!")
        return

    # PHASE 1: Build chunk_map first (để parents có sẵn khi tạo enriched embeddings)
    print("\n📦 Phase 1: Building chunk_map...")
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    vector_store["chunk_map"] = chunk_map
    print(f"✓ Đã tạo chunk_map với {len(chunk_map)} chunks")

    # PHASE 2: Generate enriched embeddings (có thể dùng parent context)
    print("\n🔮 Phase 2: Generating enriched embeddings...")
    embeddings = embed_chunks(chunks)

    # Save to vector store
    vector_store["chunks"] = chunks
    vector_store["embeddings"] = embeddings

    print("\n✅ Chatbot sẵn sàng phục vụ!")
    print(f"📊 {len(chunks)} chunks đã được load")
    print(f"🤖 Model: {config.LLM_MODEL}")
    print(f"🔍 Embedding: {config.EMBEDDING_MODEL}")
    print(f"🌟 Features: Enriched Embeddings + Smart Descendants/Siblings")
    print("="*80 + "\n")


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "Tendoo Customer Support Chatbot",
        "version": "1.0",
        "chunks_loaded": len(vector_store["chunks"]),
        "embeddings_created": len(vector_store["embeddings"]),
        "model": config.LLM_MODEL
    }


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Chat endpoint - Main API"""

    if not vector_store["chunks"]:
        raise HTTPException(status_code=500, detail="Chunks chưa được load. Hãy chạy test_tendoo.py trước.")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query không được để trống")

    print(f"\n💬 Câu hỏi: {query}")

    # 1. Tạo query embedding
    query_embedding = create_embedding(query)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Không thể tạo query embedding")

    # 2. Hybrid search
    retrieved = hybrid_search(query, query_embedding, top_k=config.TOP_K * 2)
    print(f"🔍 Tìm thấy {len(retrieved)} chunks từ hybrid search")

    # 3. Enrich with graph (smart selection)
    if config.USE_GRAPH_ENRICHMENT:
        enriched = enrich_with_graph(retrieved, query, query_embedding)
        print(f"📚 Làm giàu thành {len(enriched)} chunks (có parent/smart children/smart siblings)")
    else:
        enriched = [{"chunk": r["chunk"], "score": r["score"], **r["chunk"]["metadata"]}
                   for r in retrieved]

    # 4. Rerank (optional)
    if config.USE_RERANKING and len(enriched) > config.RERANK_TOP_K:
        # Only rerank main chunks
        main_chunks = [c for c in enriched if c.get("type") == "main"]
        reranked = rerank_with_llm(query, main_chunks)
        print(f"🎯 Rerank xuống {len(reranked)} chunks chính")

        # Re-enrich with selected chunks
        final_chunks = []
        for chunk in reranked:
            # Add main chunk
            final_chunks.append(chunk)
            # Add its context
            for c in enriched:
                if c.get("type") != "main" and c.get("chunk_id") != chunk["chunk_id"]:
                    # Check if this context belongs to current main chunk
                    final_chunks.append(c)
                    if len(final_chunks) >= 15:  # Limit total context
                        break
            if len(final_chunks) >= 15:
                break
    else:
        final_chunks = enriched[:10]

    # 5. Generate answer
    answer = generate_answer(query, final_chunks)
    print(f"✅ Đã sinh câu trả lời")

    # 6. Prepare response
    retrieved_docs = []
    for item in retrieved[:config.TOP_K]:
        chunk = item["chunk"]
        retrieved_docs.append({
            "chunk_id": chunk["chunk_id"],
            "section_code": chunk["metadata"]["section_code"],
            "section_title": chunk["metadata"]["section_title"],
            "content": chunk["content"][:300] + "..." if len(chunk["content"]) > 300 else chunk["content"],
            "score": item["score"],
            "title_path": " > ".join(chunk["metadata"]["title_path"])
        })

    return QueryResponse(
        answer=answer,
        retrieved_chunks=retrieved_docs,
        metadata={
            "total_retrieved": len(retrieved),
            "total_enriched": len(enriched),
            "total_used": len(final_chunks),
            "hybrid_search": config.USE_HYBRID_SEARCH,
            "graph_enrichment": config.USE_GRAPH_ENRICHMENT,
            "reranking": config.USE_RERANKING
        },
        conversation_id=request.conversation_id
    )


@app.get("/stats")
async def stats():
    """Thống kê về hệ thống"""
    if not vector_store["chunks"]:
        return {"error": "Chunks chưa được load"}

    chunks = vector_store["chunks"]

    # Thống kê
    type_counts = defaultdict(int)
    level_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    for chunk in chunks:
        type_counts[chunk["metadata"]["section_type"]] += 1
        level_counts[chunk["metadata"]["level"]] += 1
        for tag in chunk["metadata"]["tags"]:
            tag_counts[tag] += 1

    return {
        "total_chunks": len(chunks),
        "by_section_type": dict(type_counts),
        "by_level": dict(level_counts),
        "top_tags": dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "config": {
            "llm_model": config.LLM_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
            "hybrid_search": config.USE_HYBRID_SEARCH,
            "graph_enrichment": config.USE_GRAPH_ENRICHMENT,
            "reranking": config.USE_RERANKING
        }
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
