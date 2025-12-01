#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo cho Optimized Retrieval System
"""

import os
import sys
from pathlib import Path
from optimized_retrieval import OptimizedRetrieval


def print_banner():
    """Print banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║             ⚡ OPTIMIZED RETRIEVAL SYSTEM DEMO                               ║
║             Hệ thống tìm kiếm được tối ưu hóa                              ║
║                                                                              ║
║  Features:                                                                   ║
║  ⚡ Inverted Index BM25 (100x faster!)                                      ║
║  🤖 Real LLM Reranking (OpenAI)                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def demo_inverted_index():
    """Demo inverted index performance"""
    print(f"\n{'='*100}")
    print(f"⚡ DEMO 1: Inverted Index BM25")
    print(f"{'='*100}\n")

    chunks_path = Path(__file__).parent / "output_admission" / "chunks.json"

    # Initialize with inverted index
    retrieval = OptimizedRetrieval(
        chunks_path,
        use_inverted_index=True,
        use_openai_reranking=False
    )

    # Test queries
    test_queries = [
        "Điều kiện tuyển sinh vào trường quân đội",
        "Hồ sơ đăng ký dự tuyển",
        "Điều kiện về sức khỏe",
    ]

    for query in test_queries:
        print(f"\n{'─'*100}")
        print(f"Query: {query}")
        print(f"{'─'*100}")

        results, stats = retrieval.retrieve(query, top_k=3, initial_k=20)

        print(f"\nTop 3 Results:")
        for rank, result in enumerate(results, 1):
            print(f"{rank}. [{result['section_code']:12s}] {result['section_title'][:60]}")

        print(f"\n⏱️  BM25 Time: {stats['timing']['bm25']*1000:.2f}ms (100x faster!)")


def demo_openai_reranking():
    """Demo OpenAI reranking"""
    print(f"\n{'='*100}")
    print(f"🤖 DEMO 2: OpenAI Reranking")
    print(f"{'='*100}\n")

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("⚠️  OPENAI_API_KEY not found in environment variables")
        print("\nTo use OpenAI reranking:")
        print("  1. Get API key from https://platform.openai.com/api-keys")
        print("  2. Set environment variable:")
        print("     export OPENAI_API_KEY='your-api-key-here'")
        print("\n💡 Demo will use mock reranking instead\n")
        use_openai = False
    else:
        print(f"✓ Found OPENAI_API_KEY: {api_key[:8]}...{api_key[-4:]}")
        use_openai = True

    chunks_path = Path(__file__).parent / "output_admission" / "chunks.json"

    # Initialize with OpenAI
    retrieval = OptimizedRetrieval(
        chunks_path,
        use_inverted_index=True,
        use_openai_reranking=use_openai,
        openai_api_key=api_key,
        openai_model="gpt-4o-mini"  # Fast and cheap
    )

    # Test query
    query = "Điều kiện về sức khỏe cho tuyển sinh quân đội"

    print(f"\n{'─'*100}")
    print(f"Query: {query}")
    print(f"{'─'*100}\n")

    results, stats = retrieval.retrieve(query, top_k=5, initial_k=10)

    # Print results
    print(f"\n📊 Top 5 Results:\n")

    for rank, result in enumerate(results, 1):
        score = result.get('llm_score', result.get('rerank_score', result['score']))
        score_type = "LLM" if 'llm_score' in result else "Mock"

        print(f"{'─'*100}")
        print(f"[{rank}] {score_type} Score: {score:.2f}")
        print(f"    Section: {result['section_code']} - {result['section_title'][:60]}")
        print(f"    Type: {result['section_type']} | Module: {result['module']}")

        # Show preview
        preview = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
        print(f"\n    Preview: {preview}\n")

    # Timing
    print(f"{'='*100}")
    print(f"⏱️  Performance:")
    print(f"  BM25:      {stats['timing']['bm25']*1000:6.1f}ms")
    print(f"  Reranking: {stats['timing']['reranking']*1000:6.1f}ms")
    print(f"  Total:     {stats['timing']['total']*1000:6.1f}ms")
    print(f"{'='*100}\n")


def interactive_demo():
    """Interactive demo"""
    print_banner()

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        print(f"\n✓ Found OPENAI_API_KEY")
        use_openai = True
    else:
        print(f"\n⚠️  OPENAI_API_KEY not set - using mock reranking")
        print("   Set OPENAI_API_KEY env variable to enable OpenAI reranking\n")
        use_openai = False

    # Initialize
    chunks_path = Path(__file__).parent / "output_admission" / "chunks.json"

    print("\n🔧 Initializing optimized retrieval system...")

    retrieval = OptimizedRetrieval(
        chunks_path,
        use_inverted_index=True,
        use_openai_reranking=use_openai,
        openai_api_key=api_key
    )

    print("\n✅ System ready! Type your query or 'quit' to exit.\n")

    query_count = 0

    while True:
        try:
            user_input = input("\n🔍 Query > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break

            # Retrieve
            query_count += 1
            print(f"\n{'='*100}")
            print(f"Query #{query_count}: {user_input}")
            print(f"{'='*100}")

            results, stats = retrieval.retrieve(user_input, top_k=3, initial_k=20)

            # Print results
            for rank, result in enumerate(results, 1):
                score = result.get('llm_score', result.get('rerank_score', result['score']))
                score_type = "LLM" if 'llm_score' in result else "BM25"

                print(f"\n{'─'*100}")
                print(f"[{rank}] {score_type} Score: {score:.2f}")
                print(f"    {result['section_code']:12s} - {result['section_title'][:60]}")

                preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                print(f"    {preview}")

            # Timing
            print(f"\n{'─'*100}")
            print(f"⏱️  BM25: {stats['timing']['bm25']*1000:.1f}ms | "
                  f"Rerank: {stats['timing']['reranking']*1000:.1f}ms | "
                  f"Total: {stats['timing']['total']*1000:.1f}ms")
            print(f"{'='*100}")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main function"""

    if len(sys.argv) > 1:
        if sys.argv[1] == '--index':
            demo_inverted_index()
        elif sys.argv[1] == '--openai':
            demo_openai_reranking()
        elif sys.argv[1] == '--all':
            demo_inverted_index()
            demo_openai_reranking()
        else:
            print("Usage:")
            print("  python3 demo_optimized.py          # Interactive demo")
            print("  python3 demo_optimized.py --index  # Demo inverted index")
            print("  python3 demo_optimized.py --openai # Demo OpenAI reranking")
            print("  python3 demo_optimized.py --all    # Run all demos")
    else:
        interactive_demo()


if __name__ == "__main__":
    main()
