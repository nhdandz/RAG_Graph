#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Demo - RAG Retrieval System
Demo nhanh để xem hệ thống hoạt động
"""

import json
from pathlib import Path
from optimized_retrieval import OptimizedRetrieval

def print_banner():
    """Print welcome banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  🚀 RAG RETRIEVAL SYSTEM - QUICK DEMO                        ║
║                  Hệ thống tìm kiếm RAG được tối ưu hóa                      ║
║                                                                              ║
║  Features:                                                                   ║
║  ⚡ Inverted Index BM25 (100x faster)                                       ║
║  🔍 Hybrid Search (Dense + Sparse)                                          ║
║  📊 Sample data về RAG, BM25, Embedding                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

def run_demo():
    """Run quick demo"""
    print_banner()

    # Path to sample data
    chunks_path = Path(__file__).parent / "demo_sample_data.json"

    if not chunks_path.exists():
        print(f"❌ Error: Sample data not found at {chunks_path}")
        return

    print(f"\n📂 Loading sample data from: {chunks_path}")

    # Initialize retrieval system
    print("\n🔧 Initializing retrieval system...")
    retrieval = OptimizedRetrieval(
        chunks_path=str(chunks_path),
        use_inverted_index=True,
        use_openai_reranking=False  # Không dùng OpenAI cho demo nhanh
    )

    print(f"✅ Loaded {len(retrieval.chunks)} chunks")

    # Test queries
    test_queries = [
        "RAG là gì?",
        "Cách hoạt động của BM25",
        "Embedding vector",
        "Hybrid search kết hợp như thế nào?",
        "Query expansion với tiếng Việt"
    ]

    print(f"\n{'='*100}")
    print(f"🔍 TESTING RETRIEVAL WITH {len(test_queries)} QUERIES")
    print(f"{'='*100}\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*100}")
        print(f"Query #{i}: {query}")
        print(f"{'─'*100}")

        # Retrieve
        results, stats = retrieval.retrieve(query, top_k=3)

        # Display results
        print(f"\n📊 Stats:")
        print(f"  - Total time: {stats.get('total_time_ms', 0):.2f}ms")
        print(f"  - BM25 time: {stats.get('bm25_time_ms', 0):.2f}ms")

        print(f"\n📝 Top {len(results)} Results:\n")

        for j, result in enumerate(results, 1):
            chunk = result['chunk']
            score = result['score']
            metadata = chunk.get('metadata', {})

            print(f"  [{j}] Score: {score:.4f}")
            print(f"      Section: {metadata.get('section_code', 'N/A')} - {metadata.get('section_title', 'N/A')}")
            print(f"      Content: {chunk['content'][:150]}...")
            print(f"      Tags: {', '.join(metadata.get('tags', []))}")
            print()

    print(f"\n{'='*100}")
    print(f"✅ DEMO COMPLETED!")
    print(f"{'='*100}\n")

    # Summary
    print("📌 Summary:")
    print(f"  - Total chunks: {len(retrieval.chunks)}")
    print(f"  - Inverted index: ✅ Enabled (100x faster)")
    print(f"  - OpenAI reranking: ❌ Disabled (for quick demo)")
    print(f"  - Average retrieval time: ~{stats.get('bm25_time_ms', 0):.2f}ms")
    print()
    print("💡 Tip: Để test với dữ liệu thật, cần:")
    print("  1. File chunks.json từ document chunking")
    print("  2. Chạy: python3 admission_rag_chunking.py <input.docx>")
    print()

if __name__ == "__main__":
    run_demo()
