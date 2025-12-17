# -*- coding: utf-8 -*-
"""
RAG API cho Tendoo Documentation
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
from pathlib import Path


app = FastAPI(title="Tendoo RAG API")

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
    GEMINI_API_KEY = "AIzaSyBM2d63oq4whZNtQMxrCUd1KDtVItsSALA"
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "bge-m3"
    CHUNKS_FILE = "output_tendoo/chunks.json"

    # RAG parameters
    TOP_K = 5
    MAX_DESCENDANTS = 5
    MAX_SIBLINGS = 3
    INCLUDE_PARENT = True


config = Config()

# Configure Gemini API
genai.configure(api_key=config.GEMINI_API_KEY)


# In-memory storage
vector_store = {
    "chunks": [],
    "embeddings": [],
    "chunk_map": {}
}


# Pydantic Models
class QueryRequest(BaseModel):
    query: str
    topK: int = 5


class QueryResponse(BaseModel):
    answer: str
    retrievedDocuments: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


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
    """Tạo embedding sử dụng Ollama"""
    try:
        response = ollama.embeddings(
            model=config.EMBEDDING_MODEL,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        print(f"❌ Lỗi tạo embedding: {e}")
        return []


def embed_chunks(chunks: List[Dict]):
    """Tạo embeddings cho tất cả chunks"""
    print("\n🔄 Đang tạo embeddings cho chunks...")

    embeddings = []
    for i, chunk in enumerate(chunks):
        # Tạo text để embed (tiêu đề + nội dung)
        text_to_embed = f"{chunk['metadata']['section_title']}\n{chunk['content']}"
        embedding = create_embedding(text_to_embed)

        if embedding:
            embeddings.append(embedding)
        else:
            # Nếu lỗi, tạo zero vector
            embeddings.append([0.0] * 1024)

        if (i + 1) % 10 == 0:
            print(f"  Đã embed {i + 1}/{len(chunks)} chunks")

    print(f"✅ Đã tạo {len(embeddings)} embeddings")
    return embeddings


def retrieve_similar_chunks(query_embedding: List[float], top_k: int = 5) -> List[Dict]:
    """Tìm các chunks tương đồng nhất"""
    if not vector_store["embeddings"]:
        return []

    # Tính cosine similarity
    query_emb = np.array(query_embedding).reshape(1, -1)
    chunk_embs = np.array(vector_store["embeddings"])

    similarities = cosine_similarity(query_emb, chunk_embs)[0]

    # Lấy top K
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = vector_store["chunks"][idx]
        results.append({
            "chunk": chunk,
            "similarity": float(similarities[idx])
        })

    return results


def enrich_with_context(retrieved_chunks: List[Dict]) -> List[Dict]:
    """Làm giàu context với parent, children, siblings"""
    chunk_map = vector_store["chunk_map"]
    enriched = []

    for item in retrieved_chunks:
        chunk = item["chunk"]
        similarity = item["similarity"]

        # Main chunk
        enriched.append({
            "type": "main",
            "chunk_id": chunk["chunk_id"],
            "section_code": chunk["metadata"]["section_code"],
            "section_title": chunk["metadata"]["section_title"],
            "content": chunk["content"],
            "similarity": similarity,
            "title_path": " > ".join(chunk["metadata"]["title_path"])
        })

        # Parent
        if config.INCLUDE_PARENT and chunk["metadata"]["parent_id"]:
            parent_id = chunk["metadata"]["parent_id"]
            if parent_id in chunk_map:
                parent = chunk_map[parent_id]
                enriched.append({
                    "type": "parent",
                    "chunk_id": parent["chunk_id"],
                    "section_code": parent["metadata"]["section_code"],
                    "section_title": parent["metadata"]["section_title"],
                    "content": parent["content"][:300] + "...",  # Chỉ lấy 300 ký tự
                    "title_path": " > ".join(parent["metadata"]["title_path"])
                })

        # Children/Descendants
        children_ids = chunk["metadata"]["children_ids"][:config.MAX_DESCENDANTS]
        for child_id in children_ids:
            if child_id in chunk_map:
                child = chunk_map[child_id]
                enriched.append({
                    "type": "child",
                    "chunk_id": child["chunk_id"],
                    "section_code": child["metadata"]["section_code"],
                    "section_title": child["metadata"]["section_title"],
                    "content": child["content"],
                    "title_path": " > ".join(child["metadata"]["title_path"])
                })

        # Siblings
        sibling_ids = chunk["metadata"]["sibling_ids"][:config.MAX_SIBLINGS]
        for sibling_id in sibling_ids:
            if sibling_id in chunk_map:
                sibling = chunk_map[sibling_id]
                enriched.append({
                    "type": "sibling",
                    "chunk_id": sibling["chunk_id"],
                    "section_code": sibling["metadata"]["section_code"],
                    "section_title": sibling["metadata"]["section_title"],
                    "content": sibling["content"][:200] + "...",  # Chỉ lấy 200 ký tự
                    "title_path": " > ".join(sibling["metadata"]["title_path"])
                })

    return enriched


def generate_answer(query: str, context_chunks: List[Dict]) -> str:
    """Sinh câu trả lời sử dụng Gemini"""

    # Tạo context string
    context_parts = []
    for chunk in context_chunks:
        context_parts.append(
            f"[{chunk['type'].upper()}] {chunk['section_code']}: {chunk['section_title']}\n"
            f"Path: {chunk['title_path']}\n"
            f"Content: {chunk['content']}\n"
        )

    context_str = "\n---\n".join(context_parts)

    # Prompt
    prompt = f"""Bạn là trợ lý AI chuyên về tài liệu hướng dẫn sử dụng Tendoo App.

Dựa vào thông tin sau đây, hãy trả lời câu hỏi của người dùng một cách chi tiết và chính xác.

THÔNG TIN TÀI LIỆU:
{context_str}

CÂU HỎI: {query}

HƯỚNG DẪN TRẢ LỜI:
- Trả lời dựa trên thông tin trong tài liệu
- Nếu có các bước hướng dẫn, hãy trình bày rõ ràng theo thứ tự
- Nếu có nhiều mục liên quan, hãy liệt kê đầy đủ
- Nếu không tìm thấy thông tin, hãy nói rõ
- Sử dụng tiếng Việt rõ ràng, dễ hiểu

TRẢ LỜI:"""

    try:
        model = genai.GenerativeModel(config.LLM_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Lỗi khi sinh câu trả lời: {e}")
        return f"Xin lỗi, đã có lỗi xảy ra khi sinh câu trả lời: {str(e)}"


# ==================== API ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    """Load chunks và tạo embeddings khi khởi động"""
    print("\n" + "="*80)
    print("KHỞI ĐỘNG TENDOO RAG API")
    print("="*80)

    # Load chunks
    chunks = load_chunks()
    if not chunks:
        print("⚠️ Không có chunks nào được load!")
        return

    vector_store["chunks"] = chunks

    # Tạo chunk map
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    vector_store["chunk_map"] = chunk_map

    # Tạo embeddings
    embeddings = embed_chunks(chunks)
    vector_store["embeddings"] = embeddings

    print("\n✅ Sẵn sàng xử lý queries!")
    print("="*80 + "\n")


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "Tendoo RAG API",
        "chunks_loaded": len(vector_store["chunks"]),
        "embeddings_created": len(vector_store["embeddings"])
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query endpoint"""

    if not vector_store["chunks"]:
        raise HTTPException(status_code=500, detail="Chunks chưa được load")

    print(f"\n📝 Query: {request.query}")

    # Tạo query embedding
    query_embedding = create_embedding(request.query)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Không thể tạo query embedding")

    # Retrieve similar chunks
    retrieved = retrieve_similar_chunks(query_embedding, top_k=request.topK)
    print(f"📊 Tìm thấy {len(retrieved)} chunks tương đồng")

    # Enrich with context
    enriched = enrich_with_context(retrieved)
    print(f"📚 Làm giàu thành {len(enriched)} chunks (có parent/children/siblings)")

    # Generate answer
    answer = generate_answer(request.query, enriched)
    print(f"✅ Đã sinh câu trả lời")

    # Prepare response
    retrieved_docs = []
    for item in retrieved:
        chunk = item["chunk"]
        retrieved_docs.append({
            "chunk_id": chunk["chunk_id"],
            "section_code": chunk["metadata"]["section_code"],
            "section_title": chunk["metadata"]["section_title"],
            "content": chunk["content"][:500] + "..." if len(chunk["content"]) > 500 else chunk["content"],
            "similarity": item["similarity"],
            "title_path": " > ".join(chunk["metadata"]["title_path"])
        })

    return QueryResponse(
        answer=answer,
        retrievedDocuments=retrieved_docs,
        metadata={
            "total_retrieved": len(retrieved),
            "total_enriched": len(enriched),
            "top_k": request.topK
        }
    )


@app.get("/stats")
async def stats():
    """Thống kê về hệ thống"""
    if not vector_store["chunks"]:
        return {"error": "Chunks chưa được load"}

    chunks = vector_store["chunks"]

    # Thống kê theo section type
    from collections import defaultdict
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
        "top_tags": dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
