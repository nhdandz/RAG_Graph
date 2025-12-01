#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Retrieval System cho admission chunks
Kiểm tra xem hệ thống retrieval hoạt động như thế nào với chunks.json
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import re
from collections import Counter
import math


def load_chunks(json_path: str) -> List[Dict]:
    """Load chunks from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"✓ Loaded {len(chunks)} chunks from {json_path}")
    return chunks


class SimpleBM25:
    """Simple BM25 implementation for testing"""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize Vietnamese text"""
        if not text:
            return []
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [word for word in cleaned.split() if len(word) > 1]
        return tokens

    @staticmethod
    def calculate_idf(query_terms: List[str], documents: List[str]) -> Dict[str, float]:
        """Calculate IDF scores"""
        idf_scores = {}
        total_docs = len(documents)

        for term in query_terms:
            docs_with_term = sum(1 for doc in documents if term in SimpleBM25.tokenize(doc))
            idf = math.log((total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1.0)
            idf_scores[term] = idf

        return idf_scores

    @staticmethod
    def calculate_bm25_scores(query: str, documents: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
        """Calculate BM25 scores for all documents"""
        query_terms = SimpleBM25.tokenize(query)
        if not query_terms:
            return [0.0] * len(documents)

        # Calculate average document length
        doc_lengths = [len(SimpleBM25.tokenize(doc)) for doc in documents]
        avg_doc_length = sum(doc_lengths) / len(documents) if documents else 0

        # Calculate IDF
        idf_scores = SimpleBM25.calculate_idf(query_terms, documents)

        # Calculate BM25 score for each document
        scores = []
        for doc, doc_len in zip(documents, doc_lengths):
            doc_terms = SimpleBM25.tokenize(doc)
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


def test_retrieval(chunks: List[Dict], query: str, top_k: int = 5):
    """Test retrieval with a query"""
    print(f"\n{'='*100}")
    print(f"🔍 QUERY: {query}")
    print(f"{'='*100}\n")

    # Extract documents
    documents = [chunk['content'] for chunk in chunks]

    # Calculate BM25 scores
    bm25 = SimpleBM25()
    scores = bm25.calculate_bm25_scores(query, documents)

    # Get top K results
    top_indices = np.argsort(scores)[-top_k:][::-1]

    print(f"📊 Top {top_k} Results:\n")

    results = []
    for rank, idx in enumerate(top_indices, 1):
        chunk = chunks[idx]
        score = scores[idx]
        metadata = chunk['metadata']

        result = {
            'rank': rank,
            'score': score,
            'chunk_id': chunk['chunk_id'],
            'section_code': metadata['section_code'],
            'section_type': metadata['section_type'],
            'section_title': metadata['section_title'],
            'title_path': ' > '.join(metadata['title_path']),
            'module': metadata['module'],
            'level': metadata['level'],
            'tags': metadata['tags'],
            'content_preview': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
            'word_count': metadata['word_count']
        }
        results.append(result)

        # Print result
        print(f"{'─'*100}")
        print(f"[{rank}] Score: {score:.4f}")
        print(f"    Section: {result['section_code']} - {result['section_title']}")
        print(f"    Type: {result['section_type']} | Level: {result['level']} | Module: {result['module']}")
        print(f"    Path: {result['title_path']}")
        print(f"    Tags: {', '.join(result['tags']) if result['tags'] else 'None'}")
        print(f"    Words: {result['word_count']}")
        print(f"\n    Preview:")
        print(f"    {result['content_preview']}")
        print()

    return results


def test_hierarchy_navigation(chunks: List[Dict], chunk_id: str):
    """Test hierarchy navigation from a specific chunk"""
    print(f"\n{'='*100}")
    print(f"🌲 HIERARCHY NAVIGATION")
    print(f"{'='*100}\n")

    # Find the chunk
    chunk = next((c for c in chunks if c['chunk_id'] == chunk_id), None)
    if not chunk:
        print(f"❌ Chunk {chunk_id} not found")
        return

    metadata = chunk['metadata']

    print(f"📍 Current Chunk:")
    print(f"   ID: {chunk['chunk_id']}")
    print(f"   Section: {metadata['section_code']} - {metadata['section_title']}")
    print(f"   Type: {metadata['section_type']} | Level: {metadata['level']}")
    print(f"   Module: {metadata['module']}")
    print(f"   Tags: {', '.join(metadata['tags']) if metadata['tags'] else 'None'}")

    # Find parent
    parent_id = metadata['parent_id']
    if parent_id:
        parent = next((c for c in chunks if c['chunk_id'] == parent_id), None)
        if parent:
            print(f"\n👆 Parent:")
            print(f"   {parent['metadata']['section_code']} - {parent['metadata']['section_title']}")

    # Find children
    children_ids = metadata['children_ids']
    if children_ids:
        print(f"\n👇 Children ({len(children_ids)}):")
        for child_id in children_ids[:5]:  # Show max 5
            child = next((c for c in chunks if c['chunk_id'] == child_id), None)
            if child:
                print(f"   - {child['metadata']['section_code']} - {child['metadata']['section_title']}")
        if len(children_ids) > 5:
            print(f"   ... and {len(children_ids) - 5} more")

    # Find siblings
    sibling_ids = metadata['sibling_ids']
    if sibling_ids:
        print(f"\n👥 Siblings ({len(sibling_ids)}):")
        for sibling_id in sibling_ids[:5]:  # Show max 5
            sibling = next((c for c in chunks if c['chunk_id'] == sibling_id), None)
            if sibling:
                print(f"   - {sibling['metadata']['section_code']} - {sibling['metadata']['section_title']}")
        if len(sibling_ids) > 5:
            print(f"   ... and {len(sibling_ids) - 5} more")

    # Find related
    related_ids = metadata['related_ids']
    if related_ids:
        print(f"\n🔗 Related ({len(related_ids)}):")
        for related_id in related_ids[:5]:  # Show max 5
            related = next((c for c in chunks if c['chunk_id'] == related_id), None)
            if related:
                print(f"   - {related['metadata']['section_code']} - {related['metadata']['section_title']}")
        if len(related_ids) > 5:
            print(f"   ... and {len(related_ids) - 5} more")


def analyze_chunks_distribution(chunks: List[Dict]):
    """Analyze chunk distribution"""
    print(f"\n{'='*100}")
    print(f"📊 CHUNKS DISTRIBUTION ANALYSIS")
    print(f"{'='*100}\n")

    # Count by section type
    type_counts = Counter(chunk['metadata']['section_type'] for chunk in chunks)
    print("📁 By Section Type:")
    for section_type, count in type_counts.most_common():
        print(f"   {section_type:15s}: {count:3d} chunks")

    # Count by module
    module_counts = Counter(chunk['metadata']['module'] for chunk in chunks)
    print("\n📚 By Module:")
    for module, count in module_counts.most_common():
        print(f"   {module:30s}: {count:3d} chunks")

    # Count by level
    level_counts = Counter(chunk['metadata']['level'] for chunk in chunks)
    print("\n🔢 By Level:")
    for level in sorted(level_counts.keys()):
        count = level_counts[level]
        print(f"   Level {level}: {count:3d} chunks")

    # Tag analysis
    all_tags = []
    for chunk in chunks:
        all_tags.extend(chunk['metadata']['tags'])
    tag_counts = Counter(all_tags)
    print("\n🏷️  Top Tags:")
    for tag, count in tag_counts.most_common(10):
        print(f"   {tag:20s}: {count:3d} occurrences")

    # Word count statistics
    word_counts = [chunk['metadata']['word_count'] for chunk in chunks]
    print("\n📝 Word Count Statistics:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Min words: {min(word_counts)}")
    print(f"   Max words: {max(word_counts)}")
    print(f"   Avg words: {sum(word_counts) / len(word_counts):.1f}")
    print(f"   Total words: {sum(word_counts)}")


def main():
    """Main test function"""
    # Load chunks
    chunks_path = Path(__file__).parent / "output_admission" / "chunks.json"
    chunks = load_chunks(chunks_path)

    # Analyze distribution
    analyze_chunks_distribution(chunks)

    # Test queries
    test_queries = [
        "Điều kiện tuyển sinh vào trường quân đội",
        "Hồ sơ đăng ký dự tuyển",
        "Thời gian nộp hồ sơ",
        "Điểm thi tuyển",
        "Chế độ đào tạo",
        "Điều kiện về sức khỏe"
    ]

    for query in test_queries:
        results = test_retrieval(chunks, query, top_k=3)

    # Test hierarchy navigation
    # Use the first chunk with children
    chunk_with_children = next((c for c in chunks if c['metadata']['children_ids']), None)
    if chunk_with_children:
        test_hierarchy_navigation(chunks, chunk_with_children['chunk_id'])

    print(f"\n{'='*100}")
    print("✅ Testing completed!")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
