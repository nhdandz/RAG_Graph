"""
Enhanced RAG API cho Văn bản Tuyển sinh với Hybrid Hierarchical-Graph Chunking
Hỗ trợ:
- Load chunks từ JSON (admission_rag_chunking.py output)
- Hybrid search (Dense + BM25)
- Graph-based context enrichment (parent, children, siblings, related)
- Structured context cho LLM
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Set
import numpy as np
import google.generativeai as genai
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import math
from collections import Counter
from pathlib import Path
import os


app = FastAPI(title="Admission RAG API - Enhanced")

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
    # Gemini API
    GEMINI_API_KEY = "AIzaSyDigZIj90ppAcue0sWm85ODzB6KIu62ub8"
    LLM_MODEL = "gemini-2.5-flash"  # Gemini 2.5 Flash
    FALLBACK_LLM_MODEL = "qwen3:8b"  # Ollama fallback nếu Gemini fail
    EMBEDDING_MODEL = "bge-m3"  # Giữ nguyên Ollama embedding

    # Feature flags
    USE_HYBRID_SEARCH = True
    USE_GRAPH_ENRICHMENT = True  # Sử dụng graph relationships
    USE_DEDUPLICATION = True

    # Parameters
    DEDUP_THRESHOLD = 0.85
    BM25_K1 = 1.5
    BM25_B = 0.75
    RRF_K = 60

    # Context enrichment settings
    MAX_PARENT_LEVELS = 2  # Lấy tối đa 2 level cha
    MAX_CHILDREN = 3  # Lấy tối đa 3 con
    MAX_SIBLINGS = 2  # Lấy tối đa 2 anh em

config = Config()

# Configure Gemini API
genai.configure(api_key=config.GEMINI_API_KEY)


# In-memory storage
vector_store = {
    "chunks": [],  # List of Chunk objects từ JSON
    "embeddings": [],  # Embeddings tương ứng
    "chunk_map": {}  # Map chunk_id -> chunk for fast lookup
}


# Pydantic Models
class QueryRequest(BaseModel):
    query: str
    topK: int = 3


class ContextChunk(BaseModel):
    type: str  # "primary" | "parent_context" | "semantic_neighbor"
    heading: str
    headingPath: str
    content: str
    similarity: Optional[float] = None
    importance: Optional[float] = None
    relatedTo: Optional[str] = None
    relationshipType: Optional[str] = None  # parent, child, sibling, related


class QueryResponse(BaseModel):
    answer: str
    retrievedDocuments: List[Dict[str, Any]]
    contextStructure: Dict[str, Any]


# BM25 Implementation
class BM25:
    """BM25 sparse retrieval"""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize Vietnamese text"""
        if not text:
            return []
        # Giữ nguyên dấu tiếng Việt, chỉ xóa ký tự đặc biệt
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
            idf = math.log((total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1.0)
            idf_scores[term] = idf

        return idf_scores

    @staticmethod
    def calculate_bm25_scores(query: str, documents: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
        """Calculate BM25 scores"""
        query_terms = BM25.tokenize(query)
        if not query_terms:
            return [0.0] * len(documents)

        doc_lengths = [len(BM25.tokenize(doc)) for doc in documents]
        avg_doc_length = sum(doc_lengths) / len(documents) if documents else 0

        idf_scores = BM25.calculate_idf(query_terms, documents)

        scores = []
        for doc, doc_len in zip(documents, doc_lengths):
            doc_terms = BM25.tokenize(doc)
            term_freq = Counter(doc_terms)

            score = 0.0
            for term in query_terms:
                if term in term_freq:
                    tf = term_freq[term]
                    idf = idf_scores[term]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_length)
                    score += idf * (numerator / denominator)

            scores.append(score)

        return scores


def reciprocal_rank_fusion(dense_scores: List[float], sparse_scores: List[float], k: int = 60) -> List[float]:
    """Reciprocal Rank Fusion"""
    dense_ranks = np.argsort(dense_scores)[::-1]
    sparse_ranks = np.argsort(sparse_scores)[::-1]

    dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranks)}
    sparse_rank_map = {idx: rank for rank, idx in enumerate(sparse_ranks)}

    rrf_scores = []
    for i in range(len(dense_scores)):
        dense_rank = dense_rank_map.get(i, len(dense_scores))
        sparse_rank = sparse_rank_map.get(i, len(sparse_scores))
        rrf_score = (1.0 / (k + dense_rank)) + (1.0 / (k + sparse_rank))
        rrf_scores.append(rrf_score)

    return rrf_scores


def deduplicate_chunks(chunks: List[Dict], threshold: float = 0.85) -> List[Dict]:
    """Remove duplicate chunks using Jaccard similarity"""
    unique_chunks = []

    for chunk in chunks:
        is_duplicate = False
        chunk_words = set(chunk['content'].lower().split())

        for unique_chunk in unique_chunks:
            unique_words = set(unique_chunk['content'].lower().split())

            intersection = chunk_words & unique_words
            union = chunk_words | unique_words

            if union:
                jaccard = len(intersection) / len(union)
                if jaccard > threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_chunks.append(chunk)

    return unique_chunks


def get_embedding(text: str) -> np.ndarray:
    """Get embedding from Ollama BGE-M3 (giữ nguyên để tương thích với embeddings cũ)"""
    try:
        response = ollama.embed(model=config.EMBEDDING_MODEL, input=text)
        return np.array(response['embeddings'][0])
    except Exception as e:
        print(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")


def find_parent_chunks(chunk: Dict, max_levels: int = 2) -> List[Dict]:
    """
    Tìm các chunk cha trong hierarchy
    Sử dụng parent_id từ metadata
    """
    parents = []
    current_chunk = chunk

    for _ in range(max_levels):
        parent_id = current_chunk.get('metadata', {}).get('parent_id')
        if not parent_id or parent_id not in vector_store['chunk_map']:
            break

        parent_chunk = vector_store['chunk_map'][parent_id]
        parents.append(parent_chunk)
        current_chunk = parent_chunk

    return parents


def find_children_chunks(chunk: Dict, max_children: int = 3) -> List[Dict]:
    """
    Tìm các chunk con
    Sử dụng children_ids từ metadata
    """
    children_ids = chunk.get('metadata', {}).get('children_ids', [])
    children = []

    for child_id in children_ids[:max_children]:
        if child_id in vector_store['chunk_map']:
            children.append(vector_store['chunk_map'][child_id])

    return children


def find_all_descendants(chunk: Dict) -> List[Dict]:
    """
    Tìm TẤT CẢ con cháu của chunk (đệ quy toàn bộ cây con)
    Trả về list tất cả descendants theo thứ tự depth-first
    """
    all_descendants = []
    visited = set()  # Tránh vòng lặp

    def collect_descendants(current_chunk: Dict):
        """Đệ quy thu thập tất cả con cháu"""
        children_ids = current_chunk.get('metadata', {}).get('children_ids', [])

        for child_id in children_ids:
            if child_id in visited or child_id not in vector_store['chunk_map']:
                continue

            visited.add(child_id)
            child_chunk = vector_store['chunk_map'][child_id]
            all_descendants.append(child_chunk)

            # Đệ quy tìm con của con
            collect_descendants(child_chunk)

    collect_descendants(chunk)
    return all_descendants


def find_sibling_chunks(chunk: Dict, max_siblings: int = 2) -> List[Dict]:
    """
    Tìm các chunk anh em (cùng parent)
    Sử dụng sibling_ids từ metadata
    """
    sibling_ids = chunk.get('metadata', {}).get('sibling_ids', [])
    siblings = []

    for sibling_id in sibling_ids[:max_siblings]:
        if sibling_id in vector_store['chunk_map']:
            siblings.append(vector_store['chunk_map'][sibling_id])

    return siblings




def build_enriched_chunk_content(chunk: Dict) -> str:
    """
    Làm giàu nội dung của 1 chunk bằng cách ghép TẤT CẢ con cháu vào
    Trả về chuỗi text đầy đủ để đưa vào LLM
    """
    enriched_parts = []
    metadata = chunk['metadata']

    # 1. Thêm ngữ cảnh CHA (nếu có)
    if config.USE_GRAPH_ENRICHMENT:
        parents = find_parent_chunks(chunk, config.MAX_PARENT_LEVELS)
        if parents:
            enriched_parts.append("【 NGỮ CẢNH TỔNG QUÁT 】")
            for i, parent in enumerate(parents, 1):
                parent_title = parent['metadata'].get('section_title', 'Không rõ')
                parent_path = ' > '.join(parent['metadata'].get('title_path', []))
                enriched_parts.append(f"\n📂 Cấp cha {i}: {parent_title}")
                enriched_parts.append(f"   Vị trí: {parent_path}")
                enriched_parts.append(f"   Nội dung: {parent['content'][:300]}...")
            enriched_parts.append("")

    # 2. NỘI DUNG CHÍNH
    section_title = metadata.get('section_title', 'Không rõ')
    title_path = ' > '.join(metadata.get('title_path', []))

    enriched_parts.append("【 NỘI DUNG CHÍNH 】")
    enriched_parts.append(f"📌 Tiêu đề: {section_title}")
    enriched_parts.append(f"📍 Vị trí: {title_path}")
    enriched_parts.append(f"\n{chunk['content']}")
    enriched_parts.append("")

    # 3. Thêm TẤT CẢ CON CHÁU (đệ quy toàn bộ cây con)
    if config.USE_GRAPH_ENRICHMENT:
        all_descendants = find_all_descendants(chunk)
        if all_descendants:
            enriched_parts.append(f"【 TẤT CẢ CÁC MỤC CON ({len(all_descendants)} mục) 】")
            for i, descendant in enumerate(all_descendants, 1):
                desc_meta = descendant['metadata']
                desc_title = desc_meta.get('section_title', 'Không rõ')
                desc_path = ' > '.join(desc_meta.get('title_path', []))
                desc_level = desc_meta.get('level', 0)

                # Indent theo level để thể hiện hierarchy
                indent = "  " * (desc_level - metadata.get('level', 0))

                enriched_parts.append(f"\n{indent}🔸 [{i}] {desc_title}")
                enriched_parts.append(f"{indent}   📍 {desc_path}")
                enriched_parts.append(f"{indent}   {descendant['content']}")
            enriched_parts.append("")

    return "\n".join(enriched_parts)


def build_context_structure(primary_chunks: List[Dict]) -> List[ContextChunk]:
    """
    Chỉ trả về 1 CHUNK DUY NHẤT đã được làm giàu
    Chunk này bao gồm: parent context + nội dung chính + TẤT CẢ con cháu (đệ quy)
    """
    if not primary_chunks:
        return []

    # Chỉ lấy chunk đầu tiên (có similarity cao nhất)
    chunk = primary_chunks[0]
    metadata = chunk['metadata']
    title_path = ' > '.join(metadata.get('title_path', []))

    # Làm giàu content bằng cách ghép parent + TẤT CẢ con cháu vào
    enriched_content = build_enriched_chunk_content(chunk)

    return [ContextChunk(
        type="primary",
        heading=metadata.get('section_title', ''),
        headingPath=title_path,
        content=enriched_content,  # Content đã được làm giàu với TẤT CẢ con cháu
        similarity=chunk.get('similarity', 0.0),
        importance=1.0
    )]


def generate_answer(query: str, context_chunks: List[ContextChunk]) -> str:
    """
    Tạo câu trả lời từ LLM với 1 chunk duy nhất đã được làm giàu
    Chunk này chứa: parent context + nội dung chính + TẤT CẢ con cháu (đệ quy)
    """
    if not context_chunks:
        return "Không tìm thấy thông tin liên quan."

    # Chỉ có 1 chunk duy nhất
    chunk = context_chunks[0]

    context = f"""═══════════════════════════════════════════════════════════════
TÀI LIỆU THAM KHẢO
═══════════════════════════════════════════════════════════════

📍 Đường dẫn: {chunk.headingPath}
⭐ Độ liên quan: {chunk.similarity:.2%}

{chunk.content}
"""

    # Prompt tối ưu cho 1 chunk đầy đủ
    prompt = f"""Bạn là trợ lý ảo hỗ trợ giải đáp về Thông tư tuyển sinh của Học viện Quân sự.

NHIỆM VỤ:
1. Trả lời câu hỏi dựa trên tài liệu tham khảo bên dưới
2. Trích dẫn rõ ràng điều khoản, khoản, mục liên quan
3. Giải thích dễ hiểu, chuyên nghiệp bằng tiếng Việt

HƯỚNG DẪN:
- Tài liệu tham khảo đã bao gồm:
  • Ngữ cảnh tổng quát (phần cha)
  • Nội dung chính (phần được tìm thấy - liên quan nhất)
  • TẤT CẢ các mục con cháu (chi tiết đầy đủ)

- Sử dụng thông tin từ "Nội dung chính" và "Các mục con" để trả lời
- Nếu không đủ thông tin, nói rõ "Thông tin này không có trong tài liệu hiện có"
- Trích dẫn theo format: "Theo Điều X, Khoản Y..."
- Trả lời chi tiết, đầy đủ dựa trên toàn bộ thông tin có sẵn

════════════════════════════════════════════════════════════════
TÀI LIỆU THAM KHẢO
════════════════════════════════════════════════════════════════

{context}

════════════════════════════════════════════════════════════════
CÂU HỎI
════════════════════════════════════════════════════════════════

{query}

════════════════════════════════════════════════════════════════
TRẢ LỜI
════════════════════════════════════════════════════════════════
"""
    print(context)

    # Thử Gemini trước
    try:
        print("🔷 Trying Gemini 2.5 Flash...")
        model = genai.GenerativeModel(
            config.LLM_MODEL,
            safety_settings={
                "HARASSMENT": "BLOCK_NONE",
                "HATE_SPEECH": "BLOCK_NONE",
                "SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "DANGEROUS_CONTENT": "BLOCK_NONE",
            }
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                top_p=0.9,
                max_output_tokens=2048,
            )
        )

        # Kiểm tra response
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.finish_reason == 1:  # STOP = success
                print("✅ Gemini response successful")
                return response.text
            else:
                print(f"⚠️ Gemini blocked - finish_reason: {candidate.finish_reason}")
                print(f"⚠️ Safety ratings: {candidate.safety_ratings}")
        else:
            print("⚠️ No candidates from Gemini")

    except Exception as e:
        print(f"⚠️ Gemini error: {e}")

    # Fallback về Ollama Qwen3:8b
    try:
        print(f"🔶 Falling back to Ollama {config.FALLBACK_LLM_MODEL}...")
        response = ollama.generate(
            model=config.FALLBACK_LLM_MODEL,
            prompt=prompt,
            options={
                "temperature": 0.3,
                "top_p": 0.9,
            }
        )
        print("✅ Ollama response successful")
        return response['response']
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        raise HTTPException(status_code=500, detail=f"Both Gemini and Ollama failed. Last error: {str(e)}")


@app.post("/api/documents/load-from-json")
async def load_from_json(json_file_path: str = "output_admission/chunks.json"):
    """Load chunks từ JSON được tạo bởi admission_rag_chunking.py"""
    try:
        print(f"\n{'='*80}")
        print(f"📂 Loading chunks from: {json_file_path}")
        print(f"{'='*80}")

        # Load JSON file
        json_path = Path(__file__).parent.parent / json_file_path
        if not json_path.exists():
            json_path = Path(json_file_path)

        if not json_path.exists():
            raise HTTPException(status_code=404, detail=f"JSON file not found: {json_file_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        if not isinstance(chunks_data, list):
            raise HTTPException(status_code=400, detail="JSON must be a list of chunks")

        print(f"\n✓ Found {len(chunks_data)} chunks in JSON")

        # Clear existing data
        vector_store['chunks'].clear()
        vector_store['embeddings'].clear()
        vector_store['chunk_map'].clear()

        chunks_added = 0
        chunk_sizes = []

        for chunk_data in chunks_data:
            content = chunk_data.get('content', '').strip()
            if not content:
                continue

            metadata = chunk_data.get('metadata', {})
            word_count = metadata.get('word_count', len(content.split()))
            chunk_sizes.append(word_count)

            # Get embedding
            embedding = get_embedding(content)

            # Store chunk
            chunk_obj = {
                'chunk_id': chunk_data['chunk_id'],
                'content': content,
                'metadata': metadata,
                'similarity': 0.0  # Will be set during query
            }

            vector_store['chunks'].append(chunk_obj)
            vector_store['embeddings'].append(embedding)
            vector_store['chunk_map'][chunk_data['chunk_id']] = chunk_obj

            chunks_added += 1

            if chunks_added <= 10 or chunks_added % 20 == 0:
                section_code = metadata.get('section_code', '')
                section_title = metadata.get('section_title', '')
                print(f"  ✓ [{section_code:15s}] {section_title[:50]:50s} ({word_count:4d} words)")

        # Print statistics
        print(f"\n{'='*80}")
        print(f"📈 Loading Statistics:")
        print(f"  Chunks loaded: {chunks_added}")
        if chunk_sizes:
            print(f"  Chunk sizes (words):")
            print(f"    - Min: {min(chunk_sizes)}")
            print(f"    - Max: {max(chunk_sizes)}")
            print(f"    - Average: {sum(chunk_sizes) / len(chunk_sizes):.1f}")
        print(f"  Total chunks in store: {len(vector_store['chunks'])}")
        print(f"{'='*80}\n")

        return {
            "message": f"Loaded chunks from: {json_file_path}",
            "totalDocuments": len(vector_store['chunks']),
            "chunksAdded": chunks_added,
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


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query với Hybrid Search + Graph Enrichment"""
    try:
        if not vector_store['chunks']:
            raise HTTPException(status_code=400, detail="No documents indexed yet")

        print(f"\n{'='*80}")
        print(f"Query: {request.query}")
        print(f"{'='*80}\n")

        # Get query embedding
        query_embedding = get_embedding(request.query)

        # Stage 1: Hybrid Search
        embeddings = np.array(vector_store['embeddings'])
        dense_scores = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]

        if config.USE_HYBRID_SEARCH:
            documents_text = [chunk['content'] for chunk in vector_store['chunks']]
            sparse_scores = BM25.calculate_bm25_scores(request.query, documents_text,
                                                       config.BM25_K1, config.BM25_B)

            fused_scores = reciprocal_rank_fusion(dense_scores.tolist(), sparse_scores, config.RRF_K)
            top_indices = np.argsort(fused_scores)[-request.topK * 2:][::-1]
            print(f"Stage 1 - Hybrid search: {len(top_indices)} candidates")
        else:
            top_indices = np.argsort(dense_scores)[-request.topK * 2:][::-1]
            print(f"Stage 1 - Dense search: {len(top_indices)} candidates")

        # Get candidate chunks
        candidate_chunks = []
        for idx in top_indices:
            chunk = vector_store['chunks'][idx].copy()
            chunk['similarity'] = float(dense_scores[idx])
            candidate_chunks.append(chunk)

        # Stage 2: Deduplication
        if config.USE_DEDUPLICATION:
            candidate_chunks = deduplicate_chunks(candidate_chunks, config.DEDUP_THRESHOLD)
            print(f"Stage 2 - After dedup: {len(candidate_chunks)} chunks")

        # Get top K candidates
        final_chunks = candidate_chunks[:request.topK]
        print(f"Stage 3 - Top candidates: {len(final_chunks)} chunks")

        # Stage 4: Build 1 SINGLE enriched chunk (chỉ lấy chunk tốt nhất + TẤT CẢ con cháu)
        context_structure = build_context_structure(final_chunks)

        if context_structure:
            # Đếm số con cháu đã được thêm vào
            all_descendants = find_all_descendants(final_chunks[0]) if final_chunks else []
            print(f"\n📊 Context Structure:")
            print(f"  ✓ Chọn 1 CHUNK CHÍNH có độ liên quan cao nhất")
            print(f"  ✓ Đã ghép thêm {len(all_descendants)} mục con cháu (đệ quy)")
            print(f"  ✓ Tổng số chunks gửi cho LLM: {len(context_structure)} (1 chunk đã làm giàu)\n")

        # Generate answer
        answer = generate_answer(request.query, context_structure)

        # Prepare retrieved docs for display
        retrieved_docs = []
        for chunk in final_chunks:
            metadata = chunk['metadata']
            retrieved_docs.append({
                "filename": "Thông tư tuyển sinh",
                "content": chunk['content'],
                "similarity": chunk['similarity'],
                "section_code": metadata.get('section_code', ''),
                "section_title": metadata.get('section_title', '')
            })

        return QueryResponse(
            answer=answer,
            retrievedDocuments=retrieved_docs,
            contextStructure={
                "chunks": [chunk.dict() for chunk in context_structure],
                "summary": {
                    "primaryCount": sum(1 for c in context_structure if c.type == 'primary'),
                    "parentCount": sum(1 for c in context_structure if c.type == 'parent_context'),
                    "relatedCount": sum(1 for c in context_structure if c.type == 'semantic_neighbor'),
                    "totalCount": len(context_structure)
                }
            }
        )

    except Exception as e:
        print(f"\n❌ Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check"""
    return {
        "status": "ok",
        "service": "Admission RAG API - Enhanced with Gemini 2.5 Flash",
        "embedding_model": config.EMBEDDING_MODEL + " (Ollama)",
        "llm_model": config.LLM_MODEL,
        "llm_provider": "Google Gemini 2.5 Flash",
        "features": {
            "hybrid_search": config.USE_HYBRID_SEARCH,
            "graph_enrichment": config.USE_GRAPH_ENRICHMENT,
            "deduplication": config.USE_DEDUPLICATION
        },
        "documents": len(vector_store['chunks'])
    }


@app.get("/api/documents/count")
async def get_document_count():
    """Get total chunks"""
    return {"count": len(vector_store['chunks'])}


@app.delete("/api/documents")
async def clear_documents():
    """Clear all documents"""
    vector_store['chunks'].clear()
    vector_store['embeddings'].clear()
    vector_store['chunk_map'].clear()
    return {"message": "All documents cleared successfully"}


@app.get("/api/documents/chunks")
async def get_all_chunks(
    offset: int = 0,
    limit: int = 20,
    include_content: bool = False
):
    """
    Get all chunks with metadata

    Parameters:
    - offset: Skip first N chunks
    - limit: Max chunks to return
    - include_content: Include full content (default: preview only)
    """
    if not vector_store['chunks']:
        return {
            "total": 0,
            "showing": 0,
            "offset": 0,
            "limit": limit,
            "chunks": [],
            "summary": {}
        }

    all_chunks = vector_store['chunks']
    total = len(all_chunks)

    # Pagination
    start_idx = offset
    end_idx = min(offset + limit, total)
    paginated_chunks = all_chunks[start_idx:end_idx]

    # Format chunks for response
    formatted_chunks = []
    for idx, chunk in enumerate(paginated_chunks, start=start_idx):
        metadata = chunk['metadata']

        chunk_info = {
            "index": idx,
            "chunk_id": chunk['chunk_id'],
            "heading": metadata.get('section_title', ''),
            "headingPath": ' > '.join(metadata.get('title_path', [])),
            "level": metadata.get('level', 0),
            "section_code": metadata.get('section_code', ''),
            "section_type": metadata.get('section_type', ''),
            "module": metadata.get('module', ''),
            "wordCount": metadata.get('word_count', 0),
            "tags": metadata.get('tags', []),
            "is_global_context": metadata.get('is_global_context', False),
            "parent_id": metadata.get('parent_id'),
            "children_count": len(metadata.get('children_ids', [])),
            "siblings_count": len(metadata.get('sibling_ids', []))
        }

        if include_content:
            chunk_info['content'] = chunk['content']
        else:
            content = chunk['content']
            chunk_info['contentPreview'] = content[:200] + "..." if len(content) > 200 else content
            chunk_info['contentLength'] = len(content)

        formatted_chunks.append(chunk_info)

    # Summary statistics
    word_counts = [chunk['metadata'].get('word_count', 0) for chunk in all_chunks]
    section_types = {}
    modules = {}
    levels = {}

    for chunk in all_chunks:
        metadata = chunk['metadata']

        # Count section types
        stype = metadata.get('section_type', 'unknown')
        section_types[stype] = section_types.get(stype, 0) + 1

        # Count modules
        module = metadata.get('module', 'unknown')
        modules[module] = modules.get(module, 0) + 1

        # Count levels
        level = metadata.get('level', 0)
        levels[str(level)] = levels.get(str(level), 0) + 1

    # Get unique files (for compatibility with frontend)
    unique_files = list(set(chunk['metadata'].get('module', 'Unknown') for chunk in all_chunks))

    summary = {
        "totalChunks": total,
        "uniqueFiles": len(unique_files),
        "files": unique_files,  # List of modules (Chương I, Chương II, etc.)
        "chunkSizeStats": {
            "min": min(word_counts) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
            "avg": sum(word_counts) / len(word_counts) if word_counts else 0
        },
        "typeDistribution": section_types,  # Rename for frontend compatibility
        "levelDistribution": levels,
        "moduleDistribution": modules,  # Keep for additional info
        "globalContextCount": sum(1 for c in all_chunks if c['metadata'].get('is_global_context', False))
    }

    return {
        "total": total,
        "showing": len(formatted_chunks),
        "offset": offset,
        "limit": limit,
        "chunks": formatted_chunks,
        "summary": summary
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
