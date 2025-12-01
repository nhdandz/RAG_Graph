#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 DEMO: Enhanced Retrieval System
Interactive demo để test retrieval với custom queries
"""

from pathlib import Path
from enhanced_retrieval import EnhancedRetrieval
import sys


def print_banner():
    """Print welcome banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║             🎯 ENHANCED RETRIEVAL SYSTEM DEMO                                ║
║             Hệ thống tìm kiếm nâng cao cho tài liệu tuyển sinh             ║
║                                                                              ║
║  Features:                                                                   ║
║  ✓ Query Expansion với từ đồng nghĩa tiếng Việt                           ║
║  ✓ LLM Reranking để tăng độ chính xác                                      ║
║  ✓ Embedding Cache để tăng tốc độ                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def print_help():
    """Print help commands"""
    print("""
📖 Hướng dẫn sử dụng:

Commands:
  <query>       - Nhập câu hỏi để tìm kiếm
  examples      - Hiển thị các câu hỏi mẫu
  stats         - Xem thống kê cache
  config        - Xem cấu hình hiện tại
  toggle:exp    - Bật/tắt query expansion
  toggle:rerank - Bật/tắt reranking
  help          - Hiển thị hướng dẫn
  quit/exit     - Thoát

Ví dụ:
  > Điều kiện tuyển sinh vào trường quân đội
  > Hồ sơ đăng ký dự tuyển
  > Thời gian nộp hồ sơ
""")


def print_examples():
    """Print example queries"""
    print("""
📝 Câu hỏi mẫu:

1. Điều kiện tuyển sinh vào trường quân đội
2. Hồ sơ đăng ký dự tuyển
3. Thời gian nộp hồ sơ
4. Điều kiện về sức khỏe
5. Điểm thi tuyển
6. Chế độ đào tạo
7. Các trường tuyển sinh
8. Quy trình xét tuyển
9. Tiêu chuẩn chính trị
10. Kết quả tuyển sinh

Tip: Copy/paste một câu hỏi để test nhanh!
""")


def print_config(retrieval: EnhancedRetrieval):
    """Print current configuration"""
    print(f"""
⚙️  Cấu hình hiện tại:

  Query Expansion:  {'✅ ENABLED' if retrieval.use_expansion else '❌ DISABLED'}
  Reranking:        {'✅ ENABLED' if retrieval.use_reranking else '❌ DISABLED'}
  Cache:            {'✅ ENABLED' if retrieval.use_cache else '❌ DISABLED'}

  Chunks loaded:    {len(retrieval.chunks)}
  Documents:        {len(retrieval.documents)}
""")


def print_stats(retrieval: EnhancedRetrieval):
    """Print cache statistics"""
    if retrieval.cache:
        stats = retrieval.cache.get_stats()
        total = stats['hits'] + stats['misses']

        print(f"""
💾 Cache Statistics:

  Cache size:       {stats['size']} embeddings
  Total requests:   {total}
  Cache hits:       {stats['hits']}
  Cache misses:     {stats['misses']}
  Hit rate:         {stats['hit_rate']:.1f}%

  {'🎯 Good cache performance!' if stats['hit_rate'] > 50 else '📊 Building cache...'}
""")
    else:
        print("\n💾 Cache is disabled\n")


def format_result(result: dict, rank: int, show_preview: bool = True) -> str:
    """Format a single result"""
    output = []
    output.append(f"\n{'─'*80}")

    rerank_score = result.get('rerank_score', result['score'])
    output.append(f"[{rank}] Score: {rerank_score:.2f}")

    # Section info
    output.append(f"    📍 Section: {result['section_code']} - {result['section_title']}")
    output.append(f"    📂 Type: {result['section_type']} | Level: {result['level']} | Module: {result['module']}")

    # Tags
    if result['tags']:
        tags_str = ', '.join(result['tags'])
        output.append(f"    🏷️  Tags: {tags_str}")

    # Stats
    output.append(f"    📊 Words: {result['word_count']}")

    # Preview
    if show_preview:
        preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
        output.append(f"\n    💬 Preview:")
        output.append(f"    {preview}")

    return '\n'.join(output)


def interactive_demo():
    """Run interactive demo"""
    print_banner()

    # Initialize retrieval system
    chunks_path = Path(__file__).parent / "output_admission" / "chunks.json"

    print("\n🔧 Initializing retrieval system...")
    retrieval = EnhancedRetrieval(
        chunks_path,
        use_cache=True,
        use_expansion=True,
        use_reranking=True
    )

    print("\n✅ System ready!")
    print("\nType 'help' for commands, 'examples' for sample queries, or enter your question directly.\n")

    query_count = 0

    while True:
        try:
            # Get user input
            user_input = input("\n🔍 Query > ").strip()

            if not user_input:
                continue

            # Handle commands
            command = user_input.lower()

            if command in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Saving cache...")
                retrieval.save_cache()
                print("✅ Cache saved.\n")
                break

            elif command == 'help':
                print_help()
                continue

            elif command == 'examples':
                print_examples()
                continue

            elif command == 'stats':
                print_stats(retrieval)
                continue

            elif command == 'config':
                print_config(retrieval)
                continue

            elif command == 'toggle:exp':
                retrieval.use_expansion = not retrieval.use_expansion
                status = "ENABLED" if retrieval.use_expansion else "DISABLED"
                print(f"\n✓ Query Expansion: {status}\n")
                continue

            elif command == 'toggle:rerank':
                retrieval.use_reranking = not retrieval.use_reranking
                status = "ENABLED" if retrieval.use_reranking else "DISABLED"
                print(f"\n✓ Reranking: {status}\n")
                continue

            # Process query
            query_count += 1
            print(f"\n{'='*80}")
            print(f"Query #{query_count}: {user_input}")
            print(f"{'='*80}")

            # Retrieve
            results, stats = retrieval.retrieve(user_input, top_k=3, initial_k=20)

            # Print results
            print(f"\n📊 Results: {len(results)} chunks")

            if stats.get('expanded_queries') and len(stats['expanded_queries']) > 1:
                print(f"\n📝 Query expanded to {len(stats['expanded_queries'])} variations:")
                for i, q in enumerate(stats['expanded_queries'][:3], 1):
                    print(f"   {i}. {q}")

            for rank, result in enumerate(results, 1):
                print(format_result(result, rank, show_preview=True))

            # Performance stats
            print(f"\n{'─'*80}")
            print(f"⏱️  Performance:")
            print(f"   Total time: {stats['timing']['total']*1000:.1f}ms")
            if 'expansion' in stats['timing']:
                print(f"   - Expansion: {stats['timing']['expansion']*1000:.1f}ms")
            print(f"   - BM25: {stats['timing']['bm25']*1000:.1f}ms")
            if 'reranking' in stats['timing']:
                print(f"   - Reranking: {stats['timing']['reranking']*1000:.1f}ms")

            # Cache stats (brief)
            if retrieval.cache:
                cache_stats = retrieval.cache.get_stats()
                print(f"\n💾 Cache: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate']:.1f}% hit rate)")

            print(f"{'='*80}")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Saving cache...")
            retrieval.save_cache()
            print("✅ Cache saved.\n")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()


def batch_demo():
    """Run batch demo with predefined queries"""
    print_banner()

    chunks_path = Path(__file__).parent / "output_admission" / "chunks.json"

    print("\n🔧 Initializing retrieval system...")
    retrieval = EnhancedRetrieval(
        chunks_path,
        use_cache=True,
        use_expansion=True,
        use_reranking=True
    )

    # Test queries
    test_queries = [
        "Điều kiện tuyển sinh vào trường quân đội",
        "Hồ sơ đăng ký dự tuyển",
        "Thời gian nộp hồ sơ",
        "Điều kiện về sức khỏe",
        "Điểm thi tuyển",
    ]

    print(f"\n✅ Running batch test with {len(test_queries)} queries...\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print(f"{'='*80}")

        results, stats = retrieval.retrieve(query, top_k=3, initial_k=20)

        print(f"\nTop 3 Results:")
        for rank, result in enumerate(results[:3], 1):
            print(f"{rank}. [{result['section_code']:12s}] {result['section_title'][:60]}")

        print(f"\n⏱️  Time: {stats['timing']['total']*1000:.1f}ms")

    # Final stats
    print(f"\n{'='*80}")
    print_stats(retrieval)
    print(f"{'='*80}\n")

    retrieval.save_cache()


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        batch_demo()
    else:
        interactive_demo()


if __name__ == "__main__":
    main()
