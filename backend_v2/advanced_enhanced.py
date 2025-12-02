"""
Advanced RAG API với tất cả tính năng tối ưu:
- Query Intent Analysis & Routing
- Smart Descendants Selection (score-based)
- Multi-chunk with Smart Merging
- Adaptive Context Window
- Re-ranking with LLM
- Semantic Caching
- Query Expansion/Rewriting
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Set, Tuple
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
import hashlib
from datetime import datetime, timedelta


app = FastAPI(title="Advanced RAG API")

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
    GEMINI_API_KEY = "AIzaSyDprC2MR8frIRiVj7yzFKJGSfRFwDpXmVI"
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "bge-m3"
    # Feature flags
    USE_HYBRID_SEARCH = True
    USE_GRAPH_ENRICHMENT = True
    USE_DEDUPLICATION = True
    USE_QUERY_ANALYSIS = True
    USE_RERANKING = True
    USE_SEMANTIC_CACHE = True
    USE_QUERY_EXPANSION = True

    # Parameters
    DEDUP_THRESHOLD = 0.85
    BM25_K1 = 1.5
    BM25_B = 0.75
    RRF_K = 60

    # Advanced settings
    MAX_CHUNKS_MULTI = 3  # Số chunks tối đa trong multi-chunk mode
    MAX_SMART_DESCENDANTS = 5  # Số descendants tối đa (score-based)
    MIN_DESCENDANT_SCORE = 0.3  # Score tối thiểu để giữ descendant
    CACHE_SIMILARITY_THRESHOLD = 0.92  # Threshold cho semantic cache
    CACHE_TTL_HOURS = 24  # Cache time-to-live

    # Adaptive context settings by intent
    CONTEXT_SETTINGS = {
        "specific": {
            "chunks": 2,
            "max_descendants": 3,
            "include_parents": True,
            "include_siblings": False
        },
        "comparison": {
            "chunks": 3,
            "max_descendants": 2,
            "include_parents": True,
            "include_siblings": True
        },
        "list": {
            "chunks": 1,
            "max_descendants": 40,
            "include_parents": True,
            "include_siblings": False
        },
        "explanation": {
            "chunks": 2,
            "max_descendants": 4,
            "include_parents": True,
            "include_siblings": True
        },
        "general": {
            "chunks": 2,
            "max_descendants": 5,
            "include_parents": True,
            "include_siblings": False
        }
    }

config = Config()

# Configure Gemini API
genai.configure(api_key=config.GEMINI_API_KEY)


# In-memory storage
vector_store = {
    "chunks": [],
    "embeddings": [],
    "chunk_map": {},
    "semantic_cache": []  # List of {query_embedding, query_text, response, timestamp}
}


# Pydantic Models
class QueryRequest(BaseModel):
    query: str
    topK: int = 3


class ContextChunk(BaseModel):
    type: str
    heading: str
    headingPath: str
    content: str
    similarity: Optional[float] = None
    importance: Optional[float] = None
    relatedTo: Optional[str] = None
    relationshipType: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    retrievedDocuments: List[Dict[str, Any]]
    contextStructure: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None  # Thêm metadata về query analysis


# ==================== QUERY INTENT ANALYSIS ====================

class QueryAnalyzer:
    """Phân tích intent của câu hỏi"""

    INTENT_PATTERNS = {
        "specific": [
            r"(thời hạn|deadline|bao lâu|khi nào|ngày nào|thời gian)",
            r"(điều kiện|yêu cầu|quy định|tiêu chuẩn) (gì|nào|là gì)",
            r"(có cần|phải|bắt buộc|yêu cầu).*không",
            r"(địa chỉ|nơi|ở đâu|liên hệ)",
            r"(số lượng|bao nhiêu|mấy)"
        ],
        "comparison": [
            r"(khác nhau|khác biệt|so sánh|giống nhau)",
            r"(.*) và (.*) (khác|giống)",
            r"(chọn|lựa chọn).*(hay|hoặc)"
        ],
        "list": [
            r"(có những|bao gồm|gồm có|liệt kê|danh sách)",
            r"(các|những) (.*) (nào|gì)",
            r"(tất cả|toàn bộ|đầy đủ)"
        ],
        "explanation": [
            r"(tại sao|vì sao|lý do|nguyên nhân)",
            r"(như thế nào|thế nào|cách nào|làm sao)",
            r"(giải thích|giải|mô tả|nói rõ)",
            r"(ý nghĩa|nghĩa là gì|có nghĩa)"
        ]
    }

    @staticmethod
    def analyze(query: str) -> Dict[str, Any]:
        """
        Phân tích query và trả về intent + confidence
        """
        query_lower = query.lower()
        scores = {}

        for intent, patterns in QueryAnalyzer.INTENT_PATTERNS.items():
            score = 0
            matched_patterns = []

            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
                    matched_patterns.append(pattern)

            scores[intent] = {
                "score": score,
                "patterns": matched_patterns
            }

        # Tìm intent có score cao nhất
        if scores:
            best_intent = max(scores.items(), key=lambda x: x[1]["score"])
            if best_intent[1]["score"] > 0:
                return {
                    "intent": best_intent[0],
                    "confidence": min(best_intent[1]["score"] / 2, 1.0),
                    "matched_patterns": best_intent[1]["patterns"]
                }

        return {
            "intent": "general",
            "confidence": 0.5,
            "matched_patterns": []
        }


# ==================== QUERY EXPANSION ====================

class QueryExpander:
    """Mở rộng query để tìm kiếm tốt hơn"""

    @staticmethod
    def expand(query: str, intent: str) -> List[str]:
        """
        Tạo các biến thể của query
        """
        variations = [query]

        # Thêm từ khóa theo intent
        if intent == "specific":
            # Thêm các từ như "quy định", "theo"
            if "thời hạn" in query.lower():
                variations.append(f"{query} quy định")
                variations.append(f"thời gian {query}")

        elif intent == "list":
            # Thêm "bao gồm", "gồm có"
            variations.append(f"{query} bao gồm")
            variations.append(f"danh sách {query}")

        elif intent == "explanation":
            # Thêm "giải thích", "mô tả"
            variations.append(f"giải thích {query}")
            variations.append(f"{query} như thế nào")

        return variations[:3]  # Giới hạn 3 variations


# ==================== SEMANTIC CACHE ====================

class SemanticCache:
    """Cache các câu hỏi tương tự"""

    @staticmethod
    def get_cache_key(query: str) -> str:
        """Tạo cache key từ query"""
        return hashlib.md5(query.lower().encode()).hexdigest()

    @staticmethod
    def lookup(query_embedding: np.ndarray, threshold: float = 0.92) -> Optional[Dict]:
        """
        Tìm câu hỏi tương tự trong cache
        """
        if not vector_store['semantic_cache']:
            return None

        current_time = datetime.now()

        for cache_entry in vector_store['semantic_cache']:
            # Check TTL
            if (current_time - cache_entry['timestamp']) > timedelta(hours=config.CACHE_TTL_HOURS):
                continue

            # Check similarity
            cached_embedding = cache_entry['query_embedding']
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                cached_embedding.reshape(1, -1)
            )[0][0]

            if similarity >= threshold:
                return {
                    "response": cache_entry['response'],
                    "similarity": float(similarity),
                    "original_query": cache_entry['query_text']
                }

        return None

    @staticmethod
    def add(query_text: str, query_embedding: np.ndarray, response: Dict):
        """Thêm vào cache"""
        vector_store['semantic_cache'].append({
            "query_text": query_text,
            "query_embedding": query_embedding,
            "response": response,
            "timestamp": datetime.now()
        })

        # Giới hạn cache size
        if len(vector_store['semantic_cache']) > 100:
            vector_store['semantic_cache'] = vector_store['semantic_cache'][-100:]


# ==================== BM25 Implementation ====================

class BM25:
    """BM25 sparse retrieval"""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text:
            return []
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [word for word in cleaned.split() if len(word) > 1]
        return tokens

    @staticmethod
    def calculate_idf(query_terms: List[str], documents: List[str]) -> Dict[str, float]:
        idf_scores = {}
        total_docs = len(documents)

        for term in query_terms:
            docs_with_term = sum(1 for doc in documents if term in BM25.tokenize(doc))
            idf = math.log((total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1.0)
            idf_scores[term] = idf

        return idf_scores

    @staticmethod
    def calculate_bm25_scores(query: str, documents: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
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
    """Remove duplicate chunks"""
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
    """Get embedding from Ollama BGE-M3"""
    try:
        response = ollama.embed(model=config.EMBEDDING_MODEL, input=text)
        return np.array(response['embeddings'][0])
    except Exception as e:
        print(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")


# ==================== SMART DESCENDANTS SELECTION ====================

def score_descendant_relevance(descendant: Dict, query: str, query_embedding: np.ndarray) -> float:
    """
    Tính điểm liên quan của descendant với query
    Kết hợp: semantic similarity + keyword matching
    """
    # Semantic similarity
    desc_embedding = get_embedding(descendant['content'])
    semantic_score = cosine_similarity(
        query_embedding.reshape(1, -1),
        desc_embedding.reshape(1, -1)
    )[0][0]

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
    query_embedding: np.ndarray,
    max_descendants: int = 5,
    min_score: float = 0.3
) -> List[Tuple[Dict, float]]:
    """
    Tìm descendants có score cao nhất (thay vì lấy tất cả)
    Trả về: List of (descendant, score)
    """
    all_descendants = []
    visited = set()

    def collect_descendants(current_chunk: Dict):
        children_ids = current_chunk.get('metadata', {}).get('children_ids', [])

        for child_id in children_ids:
            if child_id in visited or child_id not in vector_store['chunk_map']:
                continue

            visited.add(child_id)
            child_chunk = vector_store['chunk_map'][child_id]
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


# ==================== GRAPH NAVIGATION ====================

def find_parent_chunks(chunk: Dict, max_levels: int = 2) -> List[Dict]:
    """Tìm các chunk cha"""
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


def find_sibling_chunks(chunk: Dict, max_siblings: int = 2) -> List[Dict]:
    """Tìm các chunk anh em"""
    sibling_ids = chunk.get('metadata', {}).get('sibling_ids', [])
    siblings = []

    for sibling_id in sibling_ids[:max_siblings]:
        if sibling_id in vector_store['chunk_map']:
            siblings.append(vector_store['chunk_map'][sibling_id])

    return siblings


# ==================== MULTI-CHUNK WITH SMART MERGING ====================

def check_hierarchy_overlap(chunk1: Dict, chunk2: Dict) -> Optional[str]:
    """
    Kiểm tra xem 2 chunks có overlap trong hierarchy không
    Trả về: "chunk1_is_ancestor" | "chunk2_is_ancestor" | None
    """
    chunk1_id = chunk1['chunk_id']
    chunk2_id = chunk2['chunk_id']

    # Check if chunk1 is ancestor of chunk2
    current = chunk2
    for _ in range(10):  # Limit depth
        parent_id = current.get('metadata', {}).get('parent_id')
        if not parent_id:
            break
        if parent_id == chunk1_id:
            return "chunk1_is_ancestor"
        if parent_id not in vector_store['chunk_map']:
            break
        current = vector_store['chunk_map'][parent_id]

    # Check if chunk2 is ancestor of chunk1
    current = chunk1
    for _ in range(10):
        parent_id = current.get('metadata', {}).get('parent_id')
        if not parent_id:
            break
        if parent_id == chunk2_id:
            return "chunk2_is_ancestor"
        if parent_id not in vector_store['chunk_map']:
            break
        current = vector_store['chunk_map'][parent_id]

    return None


def merge_chunks_smart(chunks: List[Dict], query: str, query_embedding: np.ndarray, settings: Dict) -> List[Dict]:
    """
    Merge chunks thông minh:
    - Nếu chunk A là cha của chunk B → chỉ giữ A
    - Nếu không overlap → giữ cả 2
    """
    if len(chunks) <= 1:
        return chunks

    # Check overlaps
    keep_chunks = []
    skip_indices = set()

    for i, chunk1 in enumerate(chunks):
        if i in skip_indices:
            continue

        is_redundant = False

        for j, chunk2 in enumerate(chunks):
            if i == j or j in skip_indices:
                continue

            overlap = check_hierarchy_overlap(chunk1, chunk2)

            if overlap == "chunk2_is_ancestor":
                # chunk2 là cha của chunk1 → skip chunk1, giữ chunk2
                is_redundant = True
                break
            elif overlap == "chunk1_is_ancestor":
                # chunk1 là cha của chunk2 → skip chunk2
                skip_indices.add(j)

        if not is_redundant:
            keep_chunks.append(chunk1)

    return keep_chunks


# ==================== BUILD CONTEXT WITH ADAPTIVE SETTINGS ====================

def build_enriched_context(
    chunk: Dict,
    query: str,
    query_embedding: np.ndarray,
    settings: Dict
) -> str:
    """
    Làm giàu nội dung chunk theo settings adaptive
    """
    enriched_parts = []
    metadata = chunk['metadata']

    # 1. Parent context (nếu cần)
    if settings.get('include_parents', True):
        parents = find_parent_chunks(chunk, max_levels=2)
        if parents:
            enriched_parts.append("【 NGỮ CẢNH TỔNG QUÁT 】")
            for i, parent in enumerate(parents, 1):
                parent_title = parent['metadata'].get('section_title', 'Không rõ')
                parent_path = ' > '.join(parent['metadata'].get('title_path', []))
                enriched_parts.append(f"\n📂 Cấp cha {i}: {parent_title}")
                enriched_parts.append(f"   Vị trí: {parent_path}")
                enriched_parts.append(f"   {parent['content'][:300]}...")
            enriched_parts.append("")

    # 2. Main content
    section_title = metadata.get('section_title', 'Không rõ')
    title_path = ' > '.join(metadata.get('title_path', []))

    enriched_parts.append("【 NỘI DUNG CHÍNH 】")
    enriched_parts.append(f"📌 Tiêu đề: {section_title}")
    enriched_parts.append(f"📍 Vị trí: {title_path}")
    enriched_parts.append(f"\n{chunk['content']}")
    enriched_parts.append("")

    # 3. Smart descendants (chỉ lấy những cái liên quan)
    max_desc = settings.get('max_descendants', 5)
    smart_descendants = find_smart_descendants(
        chunk, query, query_embedding,
        max_descendants=max_desc,
        min_score=config.MIN_DESCENDANT_SCORE
    )

    if smart_descendants:
        enriched_parts.append(f"【 CÁC MỤC CON LIÊN QUAN ({len(smart_descendants)}/{max_desc}) 】")
        for i, (desc, score) in enumerate(smart_descendants, 1):
            desc_meta = desc['metadata']
            desc_title = desc_meta.get('section_title', 'Không rõ')
            desc_path = ' > '.join(desc_meta.get('title_path', []))
            desc_level = desc_meta.get('level', 0)

            indent = "  " * (desc_level - metadata.get('level', 0))

            enriched_parts.append(f"\n{indent}🔸 [{i}] {desc_title} (score: {score:.2f})")
            enriched_parts.append(f"{indent}   📍 {desc_path}")
            enriched_parts.append(f"{indent}   {desc['content']}")
        enriched_parts.append("")

    # 4. Siblings (nếu cần)
    if settings.get('include_siblings', False):
        siblings = find_sibling_chunks(chunk, max_siblings=2)
        if siblings:
            enriched_parts.append(f"【 CÁC MỤC LIÊN QUAN KHÁC ({len(siblings)}) 】")
            for i, sibling in enumerate(siblings, 1):
                sib_title = sibling['metadata'].get('section_title', 'Không rõ')
                enriched_parts.append(f"\n🔹 [{i}] {sib_title}")
                enriched_parts.append(f"   {sibling['content'][:200]}...")
            enriched_parts.append("")

    return "\n".join(enriched_parts)


def build_multi_chunk_context(
    chunks: List[Dict],
    query: str,
    query_embedding: np.ndarray,
    settings: Dict
) -> List[ContextChunk]:
    """
    Build context từ nhiều chunks đã được merge
    """
    context_chunks = []

    for idx, chunk in enumerate(chunks):
        metadata = chunk['metadata']
        title_path = ' > '.join(metadata.get('title_path', []))

        # Làm giàu từng chunk
        enriched_content = build_enriched_context(chunk, query, query_embedding, settings)

        context_chunks.append(ContextChunk(
            type="primary" if idx == 0 else "secondary",
            heading=metadata.get('section_title', ''),
            headingPath=title_path,
            content=enriched_content,
            similarity=chunk.get('similarity', 0.0),
            importance=1.0 if idx == 0 else 0.8
        ))

    return context_chunks


# ==================== RE-RANKING WITH LLM ====================

def rerank_with_llm(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Re-rank chunks bằng LLM (Gemini Flash)
    """
    if len(chunks) <= top_k:
        return chunks

    # Tạo prompt cho LLM
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        title = chunk['metadata'].get('section_title', 'Không rõ')
        content_preview = chunk['content'][:300]
        chunks_text += f"\n[{i}] {title}\n{content_preview}...\n"

    prompt = f"""Bạn là hệ thống đánh giá độ liên quan của tài liệu.

CÂU HỎI: {query}

CÁC TÀI LIỆU:
{chunks_text}

NHIỆM VỤ: Xếp hạng các tài liệu từ LIÊN QUAN NHẤT đến ÍT LIÊN QUAN NHẤT.
Chỉ trả về danh sách số thứ tự, cách nhau bởi dấu phẩy (ví dụ: 2,0,4,1,3)

XẾP HẠNG:"""

    try:
        # Try Gemini first
        print("🎯 Re-ranking with Gemini...")
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=100
            ),
            request_options={"timeout": 30}  # Add 30s timeout for re-ranking
        )

        # Check finish_reason
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason != 1:  # Not STOP (normal completion)
                print(f"⚠️ Gemini re-ranking finish_reason={finish_reason}, falling back to Ollama")
                raise Exception(f"Gemini re-ranking failed with finish_reason={finish_reason}")

        # Parse ranking
        ranking_text = response.text.strip() if response.text else ""
        if not ranking_text:
            raise Exception("Empty re-ranking response from Gemini")

        indices = [int(x.strip()) for x in ranking_text.split(',') if x.strip().isdigit()]

        # Reorder chunks
        reranked = []
        for idx in indices[:top_k]:
            if 0 <= idx < len(chunks):
                reranked.append(chunks[idx])

        # Add remaining if needed
        added_ids = {c['chunk_id'] for c in reranked}
        for chunk in chunks:
            if chunk['chunk_id'] not in added_ids and len(reranked) < top_k:
                reranked.append(chunk)

        return reranked[:top_k]

    except Exception as e:
        print(f"⚠️ Gemini re-ranking error: {e}")
        print("🔄 Falling back to Ollama for re-ranking...")

        try:
            # Fallback to Ollama
            response = ollama.chat(
                model='qwen3:8b',
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.0,
                    'num_predict': 100
                }
            )

            # Parse ranking
            ranking_text = response['message']['content'].strip()
            indices = [int(x.strip()) for x in ranking_text.split(',') if x.strip().isdigit()]

            # Reorder chunks
            reranked = []
            for idx in indices[:top_k]:
                if 0 <= idx < len(chunks):
                    reranked.append(chunks[idx])

            # Add remaining if needed
            added_ids = {c['chunk_id'] for c in reranked}
            for chunk in chunks:
                if chunk['chunk_id'] not in added_ids and len(reranked) < top_k:
                    reranked.append(chunk)

            return reranked[:top_k]

        except Exception as ollama_error:
            print(f"❌ Ollama re-ranking error: {ollama_error}")
            print("⚠️ Using original order without re-ranking")
            return chunks[:top_k]


# ==================== MAIN GENERATION ====================

def generate_answer(query: str, context_chunks: List[ContextChunk], intent: str) -> str:
    """Generate answer với adaptive prompt theo intent"""
    if not context_chunks:
        return "Không tìm thấy thông tin liên quan."

    # Build context
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(f"""
═══════════════════════════════════════
TÀI LIỆU {i}/{len(context_chunks)}
═══════════════════════════════════════
📍 Đường dẫn: {chunk.headingPath}
⭐ Độ liên quan: {chunk.similarity:.2%}
🎯 Loại: {chunk.type}

{chunk.content}
""")

    context = "\n".join(context_parts)

    # Adaptive instruction theo intent
    intent_instructions = {
        "specific": "Trả lời ngắn gọn, chính xác vào đúng điểm. Trích dẫn rõ ràng điều khoản.",
        "comparison": "So sánh chi tiết các điểm giống và khác nhau. Dùng bảng nếu cần.",
        "list": "Liệt kê đầy đủ tất cả các mục. Dùng bullet points hoặc số thứ tự.",
        "explanation": "Giải thích chi tiết, rõ ràng. Có thể dùng ví dụ minh họa.",
        "general": "Trả lời đầy đủ, có cấu trúc rõ ràng."
    }

    instruction = intent_instructions.get(intent, intent_instructions["general"])

    prompt = f"""Bạn là trợ lý ảo hỗ trợ giải đáp về Thông tư tuyển sinh của Học viện Quân sự.

NHIỆM VỤ:
1. {instruction}
2. Trích dẫn rõ ràng điều khoản, khoản, mục liên quan
3. Giải thích dễ hiểu, chuyên nghiệp bằng tiếng Việt

HƯỚNG DẪN:
- Tài liệu đã được làm giàu với:
  • Ngữ cảnh tổng quát (phần cha)
  • Nội dung chính (phần liên quan nhất)
  • Các mục con liên quan (đã được lọc theo độ liên quan)
  • Các mục liên quan khác (nếu có)

- Sử dụng thông tin từ TẤT CẢ tài liệu được cung cấp
- Nếu không đủ thông tin, nói rõ "Thông tin này không có trong tài liệu"
- Trích dẫn: "Theo Điều X, Khoản Y..."

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

    try:
        # Try Gemini first
        print("🤖 Using Gemini 2.5 Flash...")
        model = genai.GenerativeModel(config.LLM_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                top_p=0.9,
                max_output_tokens=2048,
            ),
            request_options={"timeout": 60}  # Add 60s timeout
        )

        # Check finish_reason
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            # 0=FINISH_REASON_UNSPECIFIED, 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
            if finish_reason == 2:  # MAX_TOKENS
                print("⚠️ Gemini hit max tokens limit, response may be incomplete")
            elif finish_reason == 3:  # SAFETY
                print("⚠️ Gemini blocked by safety filters, falling back to Ollama")
                raise Exception("Gemini safety filter triggered")
            elif finish_reason == 5:  # OTHER
                print("⚠️ Gemini stopped for unknown reason, falling back to Ollama")
                raise Exception("Gemini stopped unexpectedly")

        # Try to get text
        if response.text:
            return response.text
        else:
            print("⚠️ Gemini returned empty response, falling back to Ollama")
            raise Exception("Empty response from Gemini")

    except Exception as e:
        print(f"⚠️ Gemini error: {e}")
        print("🔄 Falling back to Ollama Qwen3:8b...")

        try:
            # Fallback to Ollama Qwen3:8b
            response = ollama.chat(
                model='qwen3:8b',
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'num_predict': 2048
                }
            )
            return response['message']['content']
        except Exception as ollama_error:
            print(f"❌ Ollama error: {ollama_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Both LLM providers failed. Gemini: {str(e)}, Ollama: {str(ollama_error)}"
            )


# ==================== API ENDPOINTS ====================

@app.post("/api/documents/load-from-json")
async def load_from_json(json_file_path: str = "output_admission/chunks.json"):
    """Load chunks từ JSON"""
    try:
        print(f"\n{'='*80}")
        print(f"📂 Loading chunks from: {json_file_path}")
        print(f"{'='*80}")

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

        vector_store['chunks'].clear()
        vector_store['embeddings'].clear()
        vector_store['chunk_map'].clear()
        vector_store['semantic_cache'].clear()

        chunks_added = 0
        chunk_sizes = []

        for chunk_data in chunks_data:
            content = chunk_data.get('content', '').strip()
            if not content:
                continue

            metadata = chunk_data.get('metadata', {})
            word_count = metadata.get('word_count', len(content.split()))
            chunk_sizes.append(word_count)

            embedding = get_embedding(content)

            chunk_obj = {
                'chunk_id': chunk_data['chunk_id'],
                'content': content,
                'metadata': metadata,
                'similarity': 0.0
            }

            vector_store['chunks'].append(chunk_obj)
            vector_store['embeddings'].append(embedding)
            vector_store['chunk_map'][chunk_data['chunk_id']] = chunk_obj

            chunks_added += 1

            if chunks_added <= 10 or chunks_added % 20 == 0:
                section_code = metadata.get('section_code', '')
                section_title = metadata.get('section_title', '')
                print(f"  ✓ [{section_code:15s}] {section_title[:50]:50s} ({word_count:4d} words)")

        print(f"\n{'='*80}")
        print(f"✅ Loaded {chunks_added} chunks successfully")
        print(f"{'='*80}\n")

        return {
            "message": f"Loaded {chunks_added} chunks",
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
        print(f"\n❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Advanced Query với:
    - Query Intent Analysis
    - Query Expansion
    - Semantic Caching
    - Hybrid Search
    - Smart Descendants
    - Multi-chunk Smart Merging
    - Re-ranking
    - Adaptive Context
    """
    try:
        if not vector_store['chunks']:
            raise HTTPException(status_code=400, detail="No documents indexed yet")

        print(f"\n{'='*80}")
        print(f"🔍 Query: {request.query}")
        print(f"{'='*80}\n")

        # Get query embedding
        query_embedding = get_embedding(request.query)

        # Step 1: Check Semantic Cache
        if config.USE_SEMANTIC_CACHE:
            cached = SemanticCache.lookup(query_embedding, config.CACHE_SIMILARITY_THRESHOLD)
            if cached:
                print(f"✅ Cache HIT! (similarity: {cached['similarity']:.2%})")
                print(f"   Original query: {cached['original_query']}")
                response = cached['response']
                response['metadata']['from_cache'] = True
                response['metadata']['cache_similarity'] = cached['similarity']
                return QueryResponse(**response)
            print("❌ Cache MISS - proceeding with full retrieval")

        # Step 2: Query Intent Analysis
        intent_analysis = QueryAnalyzer.analyze(request.query)
        intent = intent_analysis['intent']
        print(f"\n📊 Query Analysis:")
        print(f"   Intent: {intent}")
        print(f"   Confidence: {intent_analysis['confidence']:.2%}")

        # Get adaptive settings
        settings = config.CONTEXT_SETTINGS.get(intent, config.CONTEXT_SETTINGS['general'])
        print(f"   Settings: {settings}")

        # Step 3: Query Expansion (if enabled)
        if config.USE_QUERY_EXPANSION:
            query_variations = QueryExpander.expand(request.query, intent)
            print(f"\n🔄 Query Expansion: {len(query_variations)} variations")
        else:
            query_variations = [request.query]

        # Step 4: Hybrid Search (with all query variations)
        all_candidates = []
        embeddings = np.array(vector_store['embeddings'])

        for query_var in query_variations:
            if query_var != request.query:
                var_embedding = get_embedding(query_var)
            else:
                var_embedding = query_embedding

            dense_scores = cosine_similarity(var_embedding.reshape(1, -1), embeddings)[0]

            if config.USE_HYBRID_SEARCH:
                documents_text = [chunk['content'] for chunk in vector_store['chunks']]
                sparse_scores = BM25.calculate_bm25_scores(
                    query_var, documents_text, config.BM25_K1, config.BM25_B
                )
                fused_scores = reciprocal_rank_fusion(
                    dense_scores.tolist(), sparse_scores, config.RRF_K
                )
                top_indices = np.argsort(fused_scores)[-request.topK * 3:][::-1]
            else:
                top_indices = np.argsort(dense_scores)[-request.topK * 3:][::-1]

            for idx in top_indices:
                chunk = vector_store['chunks'][idx].copy()
                chunk['similarity'] = float(dense_scores[idx])
                all_candidates.append(chunk)

        print(f"\n🔍 Stage 1 - Hybrid Search: {len(all_candidates)} candidates")

        # Step 5: Deduplication
        if config.USE_DEDUPLICATION:
            all_candidates = deduplicate_chunks(all_candidates, config.DEDUP_THRESHOLD)
            print(f"🧹 Stage 2 - After Dedup: {len(all_candidates)} chunks")

        # Step 6: Re-ranking (if enabled)
        if config.USE_RERANKING and len(all_candidates) > settings['chunks']:
            all_candidates = rerank_with_llm(request.query, all_candidates, settings['chunks'] * 2)
            print(f"🎯 Stage 3 - After Re-ranking: {len(all_candidates)} chunks")

        # Step 7: Multi-chunk Smart Merging
        num_chunks = settings['chunks']
        selected_chunks = all_candidates[:num_chunks * 2]  # Get more for merging
        merged_chunks = merge_chunks_smart(selected_chunks, request.query, query_embedding, settings)
        final_chunks = merged_chunks[:num_chunks]

        print(f"🔗 Stage 4 - After Smart Merging: {len(final_chunks)} chunks")

        # Step 8: Build Context with Adaptive Settings
        context_structure = build_multi_chunk_context(
            final_chunks, request.query, query_embedding, settings
        )

        print(f"\n📊 Context Structure:")
        print(f"   Chunks: {len(context_structure)}")
        for i, chunk in enumerate(context_structure, 1):
            # Count descendants
            if i <= len(final_chunks):
                smart_desc = find_smart_descendants(
                    final_chunks[i-1], request.query, query_embedding,
                    settings['max_descendants'], config.MIN_DESCENDANT_SCORE
                )
                print(f"   [{i}] {chunk.heading[:50]} - {len(smart_desc)} descendants")

        # Step 9: Generate Answer
        answer = generate_answer(request.query, context_structure, intent)

        # Prepare response
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

        response_data = {
            "answer": answer,
            "retrievedDocuments": retrieved_docs,
            "contextStructure": {
                "chunks": [chunk.dict() for chunk in context_structure],
                "summary": {
                    "primaryCount": sum(1 for c in context_structure if c.type == 'primary'),
                    "secondaryCount": sum(1 for c in context_structure if c.type == 'secondary'),
                    "totalCount": len(context_structure)
                }
            },
            "metadata": {
                "intent": intent,
                "intent_confidence": intent_analysis['confidence'],
                "settings_used": settings,
                "query_variations": len(query_variations),
                "from_cache": False
            }
        }

        # Add to cache
        if config.USE_SEMANTIC_CACHE:
            SemanticCache.add(request.query, query_embedding, response_data)

        return QueryResponse(**response_data)

    except Exception as e:
        print(f"\n❌ Query error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check"""
    return {
        "status": "ok",
        "service": "Advanced RAG API with Full Optimization",
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "features": {
            "query_analysis": config.USE_QUERY_ANALYSIS,
            "query_expansion": config.USE_QUERY_EXPANSION,
            "hybrid_search": config.USE_HYBRID_SEARCH,
            "graph_enrichment": config.USE_GRAPH_ENRICHMENT,
            "deduplication": config.USE_DEDUPLICATION,
            "reranking": config.USE_RERANKING,
            "semantic_cache": config.USE_SEMANTIC_CACHE
        },
        "documents": len(vector_store['chunks']),
        "cache_entries": len(vector_store['semantic_cache'])
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
    vector_store['semantic_cache'].clear()
    return {"message": "All documents cleared successfully"}


@app.get("/api/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    cache = vector_store['semantic_cache']

    if not cache:
        return {
            "total_entries": 0,
            "oldest": None,
            "newest": None
        }

    timestamps = [entry['timestamp'] for entry in cache]

    return {
        "total_entries": len(cache),
        "oldest": min(timestamps).isoformat(),
        "newest": max(timestamps).isoformat()
    }


@app.delete("/api/cache")
async def clear_cache():
    """Clear semantic cache"""
    vector_store['semantic_cache'].clear()
    return {"message": "Cache cleared successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
