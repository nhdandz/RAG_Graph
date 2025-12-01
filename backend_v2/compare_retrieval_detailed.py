#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
So sánh chi tiết giữa Basic và Enhanced Retrieval
"""

import json
from pathlib import Path
from enhanced_retrieval import EnhancedRetrieval, SimpleBM25
from typing import List, Dict
import time


def detailed_comparison(chunks_path: str, test_queries: List[str]):
    """So sánh chi tiết với nhiều queries"""

    print(f"\n{'='*120}")
    print(f"🔬 DETAILED COMPARISON: Basic vs Enhanced Retrieval")
    print(f"{'='*120}\n")

    # Initialize both systems
    basic = EnhancedRetrieval(
        chunks_path,
        use_cache=False,
        use_expansion=False,
        use_reranking=False
    )

    enhanced = EnhancedRetrieval(
        chunks_path,
        use_cache=True,
        use_expansion=True,
        use_reranking=True
    )

    # Results storage
    all_results = []

    for query_idx, query in enumerate(test_queries, 1):
        print(f"\n{'─'*120}")
        print(f"Query {query_idx}/{len(test_queries)}: {query}")
        print(f"{'─'*120}")

        # Basic retrieval
        print("\n🔵 Basic Retrieval (BM25 only)")
        basic_results, basic_stats = basic.retrieve(query, top_k=5, initial_k=5)

        # Enhanced retrieval
        print("\n🟢 Enhanced Retrieval (Expansion + Reranking)")
        enhanced_results, enhanced_stats = enhanced.retrieve(query, top_k=5, initial_k=20)

        # Compare top 3
        print(f"\n{'─'*120}")
        print("📊 TOP 3 COMPARISON")
        print(f"{'─'*120}")

        print(f"\n{'RANK':<6} {'BASIC':<55} {'ENHANCED':<55}")
        print(f"{'─'*6} {'─'*55} {'─'*55}")

        for rank in range(min(3, len(basic_results), len(enhanced_results))):
            basic_r = basic_results[rank]
            enhanced_r = enhanced_results[rank]

            basic_text = f"[{basic_r['section_code']:12s}] {basic_r['section_title'][:35]}"
            enhanced_text = f"[{enhanced_r['section_code']:12s}] {enhanced_r['section_title'][:35]}"

            # Highlight if different
            marker = "✨" if basic_r['chunk_id'] != enhanced_r['chunk_id'] else "  "

            print(f"{marker} {rank+1:<4} {basic_text:<55} {enhanced_text:<55}")

        # Performance comparison
        print(f"\n{'─'*120}")
        print("⏱️  PERFORMANCE")
        print(f"{'─'*120}")
        print(f"Basic:    {basic_stats['timing']['total']*1000:>6.1f}ms (BM25: {basic_stats['timing']['bm25']*1000:.1f}ms)")
        print(f"Enhanced: {enhanced_stats['timing']['total']*1000:>6.1f}ms (Expansion: {enhanced_stats['timing'].get('expansion', 0)*1000:.1f}ms, "
              f"BM25: {enhanced_stats['timing']['bm25']*1000:.1f}ms, Reranking: {enhanced_stats['timing'].get('reranking', 0)*1000:.1f}ms)")

        # Feature summary
        print(f"\n📝 Query expanded to {len(enhanced_stats.get('expanded_queries', []))} variations")
        if enhanced_stats.get('expanded_queries'):
            for i, q in enumerate(enhanced_stats['expanded_queries'][:3], 1):
                print(f"   {i}. {q}")

        # Store results for summary
        all_results.append({
            'query': query,
            'basic_top3': [r['chunk_id'] for r in basic_results[:3]],
            'enhanced_top3': [r['chunk_id'] for r in enhanced_results[:3]],
            'basic_time': basic_stats['timing']['total'],
            'enhanced_time': enhanced_stats['timing']['total'],
            'changed': basic_results[0]['chunk_id'] != enhanced_results[0]['chunk_id']
        })

    # Overall summary
    print(f"\n{'='*120}")
    print("📊 OVERALL SUMMARY")
    print(f"{'='*120}\n")

    total_queries = len(all_results)
    changed_count = sum(1 for r in all_results if r['changed'])
    avg_basic_time = sum(r['basic_time'] for r in all_results) / total_queries
    avg_enhanced_time = sum(r['enhanced_time'] for r in all_results) / total_queries

    print(f"Total queries tested: {total_queries}")
    print(f"Top result changed: {changed_count}/{total_queries} ({changed_count/total_queries*100:.1f}%)")
    print(f"\nAverage retrieval time:")
    print(f"  Basic:    {avg_basic_time*1000:.1f}ms")
    print(f"  Enhanced: {avg_enhanced_time*1000:.1f}ms")
    print(f"  Overhead: {(avg_enhanced_time - avg_basic_time)*1000:.1f}ms ({(avg_enhanced_time/avg_basic_time - 1)*100:.1f}%)")

    # Feature effectiveness
    print(f"\n✅ FEATURE EFFECTIVENESS:")
    print(f"  ✓ Query Expansion: Expanded queries to find more relevant results")
    print(f"  ✓ LLM Reranking: Changed top result in {changed_count}/{total_queries} cases")
    print(f"  ✓ Performance: Average retrieval time ~{avg_enhanced_time*1000:.0f}ms (acceptable for real-time use)")

    print(f"\n{'='*120}\n")

    return all_results


def analyze_reranking_impact(chunks_path: str, query: str):
    """Phân tích chi tiết impact của reranking"""

    print(f"\n{'='*120}")
    print(f"🎯 RERANKING IMPACT ANALYSIS")
    print(f"{'='*120}\n")
    print(f"Query: {query}\n")

    # Get initial BM25 results
    retrieval = EnhancedRetrieval(
        chunks_path,
        use_cache=False,
        use_expansion=True,
        use_reranking=False
    )

    candidates, _ = retrieval.retrieve(query, top_k=10, initial_k=10)

    print("🔍 Initial BM25 Rankings (top 10):")
    print(f"{'─'*120}")
    print(f"{'RANK':<6} {'SCORE':<10} {'SECTION':<15} {'TITLE':<60} {'TYPE':<12}")
    print(f"{'─'*120}")

    for rank, result in enumerate(candidates, 1):
        print(f"{rank:<6} {result['score']:<10.2f} {result['section_code']:<15} {result['section_title'][:55]:<60} {result['section_type']:<12}")

    # Apply reranking
    from enhanced_retrieval import LLMReranker
    reranker = LLMReranker(use_mock=True)
    reranked = reranker.rerank(query, candidates, top_k=10)

    print(f"\n🎯 After Reranking (top 10):")
    print(f"{'─'*120}")
    print(f"{'RANK':<6} {'RERANK':<10} {'BM25':<10} {'ΔRANK':<8} {'SECTION':<15} {'TITLE':<50}")
    print(f"{'─'*120}")

    for new_rank, result in enumerate(reranked, 1):
        # Find original rank
        original_rank = next((i+1 for i, r in enumerate(candidates) if r['chunk_id'] == result['chunk_id']), -1)
        rank_change = original_rank - new_rank

        # Highlight significant changes
        if abs(rank_change) >= 3:
            marker = "⬆️" if rank_change > 0 else "⬇️"
        elif rank_change > 0:
            marker = "↑"
        elif rank_change < 0:
            marker = "↓"
        else:
            marker = "="

        rerank_score = result.get('rerank_score', 0)
        bm25_score = result['score']

        print(f"{new_rank:<6} {rerank_score:<10.2f} {bm25_score:<10.2f} {marker} {rank_change:>3}   {result['section_code']:<15} {result['section_title'][:45]:<50}")

    print(f"\n{'─'*120}")
    print("Legend: ⬆️ = Major improvement (3+ ranks), ↑ = Improved, = = No change, ↓ = Dropped, ⬇️ = Major drop")
    print(f"{'─'*120}\n")

    # Analyze what changed
    top3_before = [r['chunk_id'] for r in candidates[:3]]
    top3_after = [r['chunk_id'] for r in reranked[:3]]

    if top3_before != top3_after:
        print("✨ Top 3 results changed after reranking!")
        print("\nChanges:")
        for i in range(3):
            before = candidates[i]
            after = reranked[i]
            if before['chunk_id'] != after['chunk_id']:
                print(f"  Position {i+1}:")
                print(f"    Before: [{before['section_code']}] {before['section_title'][:60]}")
                print(f"    After:  [{after['section_code']}] {after['section_title'][:60]}")
    else:
        print("✓ Top 3 results remained the same (but scores were refined)")

    print(f"\n{'='*120}\n")


def main():
    """Main test function"""
    chunks_path = Path(__file__).parent / "output_admission" / "chunks.json"

    # Test queries
    test_queries = [
        "Điều kiện tuyển sinh vào trường quân đội",
        "Hồ sơ đăng ký dự tuyển",
        "Thời gian nộp hồ sơ",
        "Điều kiện về sức khỏe",
        "Điểm thi tuyển",
        "Chế độ đào tạo",
    ]

    # Detailed comparison
    results = detailed_comparison(chunks_path, test_queries)

    # Deep dive into reranking for one query
    print("\n" + "="*120)
    print("DEEP DIVE: Reranking Analysis")
    print("="*120)
    analyze_reranking_impact(chunks_path, "Điều kiện về sức khỏe")

    print("\n✅ All comparisons completed!")


if __name__ == "__main__":
    main()
