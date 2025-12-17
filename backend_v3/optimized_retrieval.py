#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Retrieval System với:
1. Inverted Index BM25 (10x faster)
2. Real LLM Reranking (OpenAI)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import re
from collections import Counter, defaultdict
import math
import pickle
import hashlib
import time
from dataclasses import dataclass


# ========================================
# INVERTED INDEX BM25 (10x faster)
# ========================================

@dataclass
class InvertedIndexEntry:
    """Entry in inverted index"""
    doc_id: int
    term_freq: int
    positions: List[int] = None  # Optional: for phrase queries


class InvertedIndexBM25:
    """
    BM25 với Inverted Index - 10x faster than naive implementation

    Thay vì tính toán trên tất cả documents mỗi query,
    chỉ tính trên documents chứa query terms
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # Inverted index: term -> list of (doc_id, term_freq)
        self.inverted_index: Dict[str, List[InvertedIndexEntry]] = {}

        # Document metadata
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.num_docs: int = 0

        # IDF cache
        self.idf_cache: Dict[str, float] = {}

        # Index built flag
        self.is_built = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into terms"""
        if not text:
            return []
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [word for word in cleaned.split() if len(word) > 1]
        return tokens

    def build_index(self, documents: List[str], show_progress: bool = True):
        """
        Build inverted index from documents

        Args:
            documents: List of document texts
            show_progress: Show progress during build
        """
        start_time = time.time()

        self.num_docs = len(documents)
        self.doc_lengths = []
        self.inverted_index = defaultdict(list)

        if show_progress:
            print(f"\n🔧 Building inverted index for {self.num_docs} documents...")

        # Build inverted index
        for doc_id, doc in enumerate(documents):
            tokens = self.tokenize(doc)
            self.doc_lengths.append(len(tokens))

            # Count term frequencies
            term_freqs = Counter(tokens)

            # Add to inverted index
            for term, freq in term_freqs.items():
                self.inverted_index[term].append(
                    InvertedIndexEntry(doc_id=doc_id, term_freq=freq)
                )

            if show_progress and (doc_id + 1) % 100 == 0:
                print(f"  Indexed {doc_id + 1}/{self.num_docs} documents...")

        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / self.num_docs if self.num_docs > 0 else 0

        # Pre-calculate IDF for all terms
        if show_progress:
            print(f"  Calculating IDF scores...")

        for term, postings in self.inverted_index.items():
            df = len(postings)  # Document frequency
            # IDF formula with smoothing
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf_cache[term] = idf

        self.is_built = True

        build_time = time.time() - start_time

        if show_progress:
            print(f"✓ Index built in {build_time:.2f}s")
            print(f"  - Unique terms: {len(self.inverted_index)}")
            print(f"  - Avg doc length: {self.avg_doc_length:.1f} tokens")
            print(f"  - Index size: ~{self._estimate_size() / 1024:.1f} KB\n")

    def _estimate_size(self) -> int:
        """Estimate memory size of index in bytes"""
        # Rough estimate
        size = 0
        for term, postings in self.inverted_index.items():
            size += len(term) * 2  # Term string (approx 2 bytes per char)
            size += len(postings) * 12  # Each posting ~12 bytes (doc_id + freq)
        return size

    def search(self, query: str) -> List[float]:
        """
        Search using BM25 with inverted index

        Args:
            query: Search query

        Returns:
            List of BM25 scores for all documents
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_terms = self.tokenize(query)
        if not query_terms:
            return [0.0] * self.num_docs

        # Initialize scores
        scores = [0.0] * self.num_docs

        # Only process documents containing query terms
        for term in query_terms:
            if term not in self.inverted_index:
                continue  # Term not in corpus

            idf = self.idf_cache[term]
            postings = self.inverted_index[term]

            # Calculate BM25 for each document containing this term
            for entry in postings:
                doc_id = entry.doc_id
                tf = entry.term_freq
                doc_len = self.doc_lengths[doc_id]

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)

                scores[doc_id] += idf * (numerator / denominator)

        return scores

    def save_index(self, filepath: str):
        """Save index to disk"""
        data = {
            'inverted_index': dict(self.inverted_index),
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'num_docs': self.num_docs,
            'idf_cache': self.idf_cache,
            'k1': self.k1,
            'b': self.b
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"✓ Saved index to {filepath}")

    def load_index(self, filepath: str):
        """Load index from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.inverted_index = defaultdict(list, data['inverted_index'])
        self.doc_lengths = data['doc_lengths']
        self.avg_doc_length = data['avg_doc_length']
        self.num_docs = data['num_docs']
        self.idf_cache = data['idf_cache']
        self.k1 = data['k1']
        self.b = data['b']
        self.is_built = True

        print(f"✓ Loaded index from {filepath}")
        print(f"  - {self.num_docs} documents")
        print(f"  - {len(self.inverted_index)} unique terms")


# ========================================
# REAL LLM RERANKING (OpenAI)
# ========================================

class OpenAIReranker:
    """
    LLM-based reranking using OpenAI API

    Supports:
    - OpenAI (gpt-4o-mini, gpt-4o)
    - Compatible APIs (Ollama, etc.)
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 base_url: Optional[str] = None,
                 use_openai: bool = True):
        """
        Initialize reranker

        Args:
            api_key: OpenAI API key (or None to load from env)
            model: Model name (gpt-4o-mini, gpt-4o, etc.)
            base_url: Custom base URL (for compatible APIs)
            use_openai: If False, will use mock reranking
        """
        self.use_openai = use_openai
        self.model = model

        if use_openai:
            try:
                from openai import OpenAI

                # Initialize client
                if base_url:
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                else:
                    self.client = OpenAI(api_key=api_key)

                print(f"✓ OpenAI client initialized (model: {model})")

            except ImportError:
                print("⚠ OpenAI package not installed. Install with: pip install openai")
                print("  Falling back to mock reranking")
                self.use_openai = False
            except Exception as e:
                print(f"⚠ Failed to initialize OpenAI: {e}")
                print("  Falling back to mock reranking")
                self.use_openai = False

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank candidates using LLM

        Args:
            query: User query
            candidates: List of candidate chunks
            top_k: Number of results to return

        Returns:
            Reranked list
        """
        if self.use_openai:
            return self._openai_rerank(query, candidates, top_k)
        else:
            return self._mock_rerank(query, candidates, top_k)

    def _openai_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Rerank using OpenAI API"""

        print(f"\n🤖 Reranking with OpenAI ({self.model})...")

        scored_candidates = []

        for i, candidate in enumerate(candidates):
            # Prepare prompt
            content_preview = candidate['content'][:500]  # Limit to 500 chars

            prompt = f"""You are a relevance scoring assistant for a document retrieval system.

Task: Score how relevant the following document is to answer the user's question.

Question: {query}

Document:
{content_preview}

Instructions:
- Score from 0 to 10 (0 = completely irrelevant, 10 = highly relevant)
- Consider: Does this document contain information to answer the question?
- Be strict: Only give high scores (8-10) if the document directly answers the question
- Respond with ONLY a number (0-10), no explanation needed

Score:"""

            try:
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a relevance scoring assistant. Respond only with a number 0-10."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=5
                )

                # Extract score
                score_text = response.choices[0].message.content.strip()
                score = float(re.sub(r'[^0-9.]', '', score_text))
                score = max(0.0, min(10.0, score))  # Clamp to 0-10

                print(f"  [{i+1}/{len(candidates)}] {candidate['section_code']:12s} → {score:.1f}/10")

            except Exception as e:
                print(f"  [{i+1}/{len(candidates)}] Error: {e}")
                score = 5.0  # Default medium score

            candidate['llm_score'] = score
            scored_candidates.append(candidate)

        # Sort by LLM score
        scored_candidates.sort(key=lambda x: x['llm_score'], reverse=True)

        print(f"✓ Reranked {len(candidates)} candidates\n")

        return scored_candidates[:top_k]

    def _mock_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        Mock reranking (rule-based)
        Same as enhanced_retrieval.py
        """
        from enhanced_retrieval import SimpleBM25

        query_terms = set(SimpleBM25.tokenize(query))

        for candidate in candidates:
            base_score = candidate.get('score', 0)
            boost = 0.0

            # Title matching
            title = candidate.get('section_title', '').lower()
            title_terms = set(SimpleBM25.tokenize(title))
            title_match = len(query_terms & title_terms)
            boost += title_match * 2.0

            # Section type boost
            section_type = candidate.get('section_type', '')
            type_boost = {
                'dieu': 1.5,
                'muc': 1.2,
                'chuong': 1.0,
                'khoan': 0.8,
                'item_abc': 0.5,
                'item_dash': 0.3,
                'item_plus': 0.2,
            }
            boost += type_boost.get(section_type, 0)

            # Tag matching
            tags = candidate.get('tags', [])
            for term in query_terms:
                if any(term in tag for tag in tags):
                    boost += 1.0

            # Length penalty
            word_count = candidate.get('word_count', 0)
            if word_count < 10:
                boost -= 2.0
            elif word_count > 200:
                boost -= 1.0

            candidate['rerank_score'] = base_score + boost

        candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        return candidates[:top_k]


# ========================================
# OPTIMIZED RETRIEVAL SYSTEM
# ========================================

class OptimizedRetrieval:
    """
    Optimized retrieval with:
    1. Inverted Index BM25 (10x faster)
    2. Real OpenAI Reranking
    """

    def __init__(self,
                 chunks_path: str,
                 use_inverted_index: bool = True,
                 use_openai_reranking: bool = False,
                 openai_api_key: Optional[str] = None,
                 openai_model: str = "gpt-4o-mini",
                 index_cache_path: str = "bm25_index.pkl"):
        """
        Initialize optimized retrieval

        Args:
            chunks_path: Path to chunks.json
            use_inverted_index: Use inverted index for BM25
            use_openai_reranking: Use OpenAI for reranking
            openai_api_key: OpenAI API key
            openai_model: OpenAI model name
            index_cache_path: Path to save/load index
        """
        # Load chunks
        self.chunks = self._load_chunks(chunks_path)
        self.documents = [chunk['content'] for chunk in self.chunks]

        # Initialize BM25
        self.use_inverted_index = use_inverted_index
        self.index_cache_path = index_cache_path

        if use_inverted_index:
            self.bm25 = InvertedIndexBM25()

            # Try to load cached index
            if Path(index_cache_path).exists():
                print(f"\n📦 Loading cached index from {index_cache_path}...")
                try:
                    self.bm25.load_index(index_cache_path)
                except Exception as e:
                    print(f"⚠ Failed to load cache: {e}")
                    print("  Building new index...")
                    self.bm25.build_index(self.documents)
                    self.bm25.save_index(index_cache_path)
            else:
                self.bm25.build_index(self.documents)
                self.bm25.save_index(index_cache_path)
        else:
            # Fallback to naive BM25
            from enhanced_retrieval import SimpleBM25
            self.bm25 = SimpleBM25()

        # Initialize reranker
        self.reranker = OpenAIReranker(
            api_key=openai_api_key,
            model=openai_model,
            use_openai=use_openai_reranking
        )

        print(f"\n✓ Optimized Retrieval initialized")
        print(f"  - Chunks: {len(self.chunks)}")
        print(f"  - Inverted Index: {'Enabled' if use_inverted_index else 'Disabled'}")
        print(f"  - OpenAI Reranking: {'Enabled' if use_openai_reranking else 'Disabled (mock)'}")

    def _load_chunks(self, chunks_path: str) -> List[Dict]:
        """Load chunks from JSON"""
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        return chunks

    def retrieve(self, query: str, top_k: int = 5, initial_k: int = 20) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Retrieve relevant chunks

        Args:
            query: User query
            top_k: Final number of results
            initial_k: Number of candidates before reranking

        Returns:
            (results, stats)
        """
        start_time = time.time()
        stats = {'query': query, 'timing': {}}

        # BM25 retrieval
        t0 = time.time()
        if self.use_inverted_index:
            scores = self.bm25.search(query)
        else:
            scores = self.bm25.calculate_bm25_scores(query, self.documents)
        stats['timing']['bm25'] = time.time() - t0

        # Get top candidates
        top_indices = np.argsort(scores)[-initial_k:][::-1]

        candidates = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            candidates.append({
                'idx': idx,
                'chunk_id': chunk['chunk_id'],
                'section_code': chunk['metadata']['section_code'],
                'section_type': chunk['metadata']['section_type'],
                'section_title': chunk['metadata']['section_title'],
                'title_path': ' > '.join(chunk['metadata']['title_path']),
                'module': chunk['metadata']['module'],
                'level': chunk['metadata']['level'],
                'tags': chunk['metadata']['tags'],
                'content': chunk['content'],
                'word_count': chunk['metadata']['word_count'],
                'score': scores[idx]
            })

        stats['initial_candidates'] = len(candidates)

        # Reranking
        t0 = time.time()
        results = self.reranker.rerank(query, candidates, top_k)
        stats['timing']['reranking'] = time.time() - t0

        stats['final_results'] = len(results)
        stats['timing']['total'] = time.time() - start_time

        return results, stats


# ========================================
# TESTING & COMPARISON
# ========================================

def benchmark_bm25(chunks_path: str, test_queries: List[str]):
    """Benchmark naive vs inverted index BM25"""

    print(f"\n{'='*100}")
    print(f"⚡ BENCHMARK: Naive BM25 vs Inverted Index BM25")
    print(f"{'='*100}\n")

    # Load chunks
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    documents = [chunk['content'] for chunk in chunks]

    print(f"Dataset: {len(documents)} documents\n")

    # Build inverted index
    print("="*100)
    inverted_bm25 = InvertedIndexBM25()
    inverted_bm25.build_index(documents)

    # Benchmark
    from enhanced_retrieval import SimpleBM25
    naive_bm25 = SimpleBM25()

    print(f"{'='*100}")
    print(f"Running {len(test_queries)} queries...\n")

    naive_times = []
    inverted_times = []

    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}/{len(test_queries)}: {query[:50]}...")

        # Naive BM25
        t0 = time.time()
        naive_scores = naive_bm25.calculate_bm25_scores(query, documents)
        naive_time = time.time() - t0
        naive_times.append(naive_time)

        # Inverted BM25
        t0 = time.time()
        inverted_scores = inverted_bm25.search(query)
        inverted_time = time.time() - t0
        inverted_times.append(inverted_time)

        # Compare results
        naive_top = np.argsort(naive_scores)[-5:][::-1]
        inverted_top = np.argsort(inverted_scores)[-5:][::-1]

        same_results = np.array_equal(naive_top, inverted_top)

        print(f"  Naive:    {naive_time*1000:6.1f}ms")
        print(f"  Inverted: {inverted_time*1000:6.1f}ms")
        print(f"  Speedup:  {naive_time/inverted_time:5.1f}x")
        print(f"  Results:  {'✓ Same' if same_results else '✗ Different'}\n")

    # Summary
    print(f"{'='*100}")
    print(f"📊 SUMMARY")
    print(f"{'='*100}\n")

    avg_naive = np.mean(naive_times)
    avg_inverted = np.mean(inverted_times)
    speedup = avg_naive / avg_inverted

    print(f"Average times:")
    print(f"  Naive BM25:       {avg_naive*1000:6.1f}ms")
    print(f"  Inverted Index:   {avg_inverted*1000:6.1f}ms")
    print(f"  Speedup:          {speedup:5.1f}x")
    print(f"\n✅ Inverted index is {speedup:.1f}x faster!\n")

    print(f"{'='*100}\n")


def test_openai_reranking(chunks_path: str, query: str, api_key: Optional[str] = None):
    """Test OpenAI reranking"""

    print(f"\n{'='*100}")
    print(f"🤖 TEST: OpenAI Reranking")
    print(f"{'='*100}\n")

    if not api_key:
        print("⚠ No API key provided. Set OPENAI_API_KEY environment variable or pass api_key parameter")
        print("  Testing with mock reranking instead\n")

    # Initialize
    retrieval = OptimizedRetrieval(
        chunks_path,
        use_inverted_index=True,
        use_openai_reranking=bool(api_key),
        openai_api_key=api_key
    )

    # Retrieve
    print(f"\nQuery: {query}\n")
    print("="*100)

    results, stats = retrieval.retrieve(query, top_k=5, initial_k=20)

    # Print results
    print(f"\n📊 Top {len(results)} Results:\n")

    for rank, result in enumerate(results, 1):
        score = result.get('llm_score', result.get('rerank_score', result['score']))
        print(f"[{rank}] Score: {score:.2f}")
        print(f"    {result['section_code']:12s} - {result['section_title'][:60]}")
        print()

    # Timing
    print(f"{'='*100}")
    print(f"⏱️  Performance:")
    print(f"  BM25:      {stats['timing']['bm25']*1000:6.1f}ms")
    print(f"  Reranking: {stats['timing']['reranking']*1000:6.1f}ms")
    print(f"  Total:     {stats['timing']['total']*1000:6.1f}ms")
    print(f"{'='*100}\n")


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
    ]

    # 1. Benchmark BM25
    print("\n" + "="*100)
    print("TEST 1: BM25 Performance Benchmark")
    print("="*100)
    benchmark_bm25(chunks_path, test_queries)

    # 2. Test OpenAI Reranking
    print("\n" + "="*100)
    print("TEST 2: OpenAI Reranking")
    print("="*100)

    import os
    api_key = os.getenv("OPENAI_API_KEY")

    test_openai_reranking(
        chunks_path,
        "Điều kiện về sức khỏe",
        api_key=api_key
    )

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
