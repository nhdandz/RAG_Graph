#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Demo - RAG BM25 Retrieval
Demo đơn giản không cần dependencies phức tạp
"""

import json
import re
import math
from collections import Counter
from typing import List, Dict, Tuple

def tokenize(text: str) -> List[str]:
    """Tokenize Vietnamese text"""
    if not text:
        return []
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = [word for word in cleaned.split() if len(word) > 1]
    return tokens

def calculate_bm25_scores(query: str, documents: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Calculate BM25 scores for documents"""
    query_terms = tokenize(query)
    if not query_terms:
        return [0.0] * len(documents)

    # Calculate document lengths
    doc_lengths = [len(tokenize(doc)) for doc in documents]
    avg_doc_length = sum(doc_lengths) / len(documents) if documents else 0

    # Calculate IDF scores
    idf_scores = {}
    for term in query_terms:
        docs_with_term = sum(1 for doc in documents if term in tokenize(doc))
        idf = math.log((len(documents) - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1.0)
        idf_scores[term] = idf

    # Calculate BM25 scores
    scores = []
    for doc, doc_len in zip(documents, doc_lengths):
        doc_tokens = tokenize(doc)
        term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_terms:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = idf_scores[term]

            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_length))
            score += idf * (numerator / denominator)

        scores.append(score)

    return scores

def retrieve(query: str, chunks: List[Dict], top_k: int = 3) -> List[Tuple[Dict, float]]:
    """Retrieve top-k relevant chunks"""
    documents = [chunk['content'] for chunk in chunks]
    scores = calculate_bm25_scores(query, documents)

    # Sort by score descending
    results = [(chunk, score) for chunk, score in zip(chunks, scores)]
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]

def print_banner():
    """Print welcome banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  🚀 SIMPLE RAG RETRIEVAL DEMO                                ║
║                  Demo đơn giản với BM25                                     ║
║                                                                              ║
║  Features:                                                                   ║
║  🔍 BM25 Search                                                             ║
║  📊 Sample data về RAG, Embedding, BM25                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

def run_demo():
    """Run simple demo"""
    print_banner()

    # Load sample data
    data_file = "demo_sample_data.json"
    print(f"\n📂 Loading data from: {data_file}")

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"✅ Loaded {len(chunks)} chunks\n")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Test queries
    test_queries = [
        "RAG là gì?",
        "BM25 hoạt động như thế nào?",
        "Embedding vector",
        "Hybrid search",
        "Query expansion tiếng Việt"
    ]

    print(f"{'='*100}")
    print(f"🔍 TESTING WITH {len(test_queries)} QUERIES")
    print(f"{'='*100}\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*100}")
        print(f"📌 Query #{i}: \"{query}\"")
        print(f"{'─'*100}\n")

        # Retrieve
        results = retrieve(query, chunks, top_k=3)

        # Display results
        for j, (chunk, score) in enumerate(results, 1):
            metadata = chunk.get('metadata', {})

            print(f"  [{j}] Score: {score:.4f}")
            print(f"      ID: {chunk.get('chunk_id', 'N/A')}")
            print(f"      Section: {metadata.get('section_code', 'N/A')} - {metadata.get('section_title', 'N/A')}")
            print(f"      Tags: {', '.join(metadata.get('tags', []))}")

            content = chunk['content']
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"      Content: {content}")
            print()

    print(f"\n{'='*100}")
    print(f"✅ DEMO HOÀN THÀNH!")
    print(f"{'='*100}\n")

    print("💡 Giải thích:")
    print("  - BM25: Thuật toán ranking dựa trên tần suất từ và IDF")
    print("  - Score cao = liên quan nhiều với query")
    print("  - Kết quả được sắp xếp theo score giảm dần")
    print()

if __name__ == "__main__":
    run_demo()
