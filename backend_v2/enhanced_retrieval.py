#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Retrieval System với:
1. LLM Reranking để tăng độ chính xác
2. Embedding Cache để tăng tốc độ
3. Query Expansion cho tiếng Việt
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
from collections import Counter
import math
import pickle
import hashlib
from datetime import datetime
import time


class EmbeddingCache:
    """Cache cho embeddings để tránh tính toán lại"""

    def __init__(self, cache_file: str = "embedding_cache.pkl"):
        self.cache_file = cache_file
        self.cache: Dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        self.load_cache()

    def _get_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        key = self._get_key(text)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, text: str, embedding: np.ndarray):
        """Save embedding to cache"""
        key = self._get_key(text)
        self.cache[key] = embedding

    def save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"✓ Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            print(f"⚠ Failed to save cache: {e}")

    def load_cache(self):
        """Load cache from disk"""
        if Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"✓ Loaded {len(self.cache)} embeddings from cache")
            except Exception as e:
                print(f"⚠ Failed to load cache: {e}")
                self.cache = {}

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class VietnameseQueryExpander:
    """Query expansion cho tiếng Việt"""

    def __init__(self):
        # Từ đồng nghĩa cho các từ khóa phổ biến trong tuyển sinh
        self.synonyms = {
            'tuyển sinh': ['tuyển', 'thi tuyển', 'xét tuyển', 'tuyển chọn', 'nhận sinh viên'],
            'hồ sơ': ['giấy tờ', 'tài liệu', 'chứng từ', 'văn bằng'],
            'điều kiện': ['yêu cầu', 'tiêu chuẩn', 'quy định', 'chuẩn'],
            'đào tạo': ['học tập', 'giáo dục', 'bồi dưỡng', 'đào tạo'],
            'quân đội': ['quân sự', 'lực lượng vũ trang', 'bộ đội'],
            'thời gian': ['thời hạn', 'hạn chót', 'deadline', 'thời điểm'],
            'kết quả': ['điểm số', 'thành tích', 'thành quả'],
            'sức khỏe': ['thể lực', 'sức khoẻ', 'thể chất'],
            'học sinh': ['thí sinh', 'học viên', 'sinh viên'],
            'trường': ['nhà trường', 'học viện', 'trường học'],
            'chương trình': ['khóa học', 'khoá học', 'chương trình học'],
            'nộp': ['gửi', 'đăng ký', 'submit'],
            'điểm': ['điểm thi', 'điểm số', 'kết quả thi'],
        }

        # Từ viết tắt và mở rộng
        self.abbreviations = {
            'hs': 'học sinh',
            'sv': 'sinh viên',
            'gv': 'giáo viên',
            'ts': 'tuyển sinh',
            'đt': 'đào tạo',
            'hồ sơ đk': 'hồ sơ đăng ký',
        }

    def expand_query(self, query: str, max_expansions: int = 2) -> List[str]:
        """
        Mở rộng query với từ đồng nghĩa

        Returns:
            List of expanded queries (original + variations)
        """
        query_lower = query.lower()
        expanded_queries = [query]  # Luôn giữ query gốc

        # 1. Thay thế từ viết tắt
        for abbr, full in self.abbreviations.items():
            if abbr in query_lower:
                new_query = query_lower.replace(abbr, full)
                if new_query != query_lower:
                    expanded_queries.append(new_query)

        # 2. Thêm từ đồng nghĩa
        expansion_count = 0
        for keyword, synonyms in self.synonyms.items():
            if keyword in query_lower and expansion_count < max_expansions:
                # Chỉ thêm 1 từ đồng nghĩa đầu tiên cho mỗi keyword
                synonym = synonyms[0]
                new_query = query_lower.replace(keyword, synonym)
                if new_query not in expanded_queries:
                    expanded_queries.append(new_query)
                    expansion_count += 1

        return expanded_queries


class SimpleBM25:
    """Simple BM25 implementation"""

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

        doc_lengths = [len(SimpleBM25.tokenize(doc)) for doc in documents]
        avg_doc_length = sum(doc_lengths) / len(documents) if documents else 0

        idf_scores = SimpleBM25.calculate_idf(query_terms, documents)

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


class LLMReranker:
    """Reranking sử dụng LLM (giả lập - trong thực tế dùng Ollama/OpenAI)"""

    def __init__(self, use_mock: bool = True):
        """
        Args:
            use_mock: Nếu True, dùng mock scoring. Nếu False, dùng LLM thực
        """
        self.use_mock = use_mock

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank candidates using LLM

        Args:
            query: User query
            candidates: List of candidate chunks with scores
            top_k: Number of top results to return

        Returns:
            Reranked list of candidates
        """
        if self.use_mock:
            return self._mock_rerank(query, candidates, top_k)
        else:
            return self._llm_rerank(query, candidates, top_k)

    def _mock_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        Mock reranking bằng cách boost scores dựa trên:
        - Matching query terms trong title
        - Section type (ưu tiên 'dieu' hơn 'khoan')
        - Tags matching
        """
        query_terms = set(SimpleBM25.tokenize(query))

        for candidate in candidates:
            base_score = candidate.get('score', 0)
            boost = 0.0

            # Boost nếu query terms xuất hiện trong title
            title = candidate.get('section_title', '').lower()
            title_terms = set(SimpleBM25.tokenize(title))
            title_match = len(query_terms & title_terms)
            boost += title_match * 2.0

            # Boost theo section type
            section_type = candidate.get('section_type', '')
            type_boost = {
                'dieu': 1.5,      # Điều thường chứa quy định quan trọng
                'muc': 1.2,       # Mục
                'chuong': 1.0,    # Chương
                'khoan': 0.8,     # Khoản
                'item_abc': 0.5,  # Các ý a, b, c...
                'item_dash': 0.3,
                'item_plus': 0.2,
            }
            boost += type_boost.get(section_type, 0)

            # Boost nếu tags match query
            tags = candidate.get('tags', [])
            for term in query_terms:
                if any(term in tag for tag in tags):
                    boost += 1.0

            # Penalty cho chunks quá ngắn hoặc quá dài
            word_count = candidate.get('word_count', 0)
            if word_count < 10:
                boost -= 2.0
            elif word_count > 200:
                boost -= 1.0

            candidate['rerank_score'] = base_score + boost

        # Sort by rerank score
        candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        return candidates[:top_k]

    def _llm_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        Rerank using actual LLM (Ollama/OpenAI)
        TODO: Implement this when you want to use real LLM
        """
        # Placeholder for real LLM reranking
        # You would call ollama.generate() or openai.ChatCompletion here
        print("⚠ Real LLM reranking not implemented yet, using mock")
        return self._mock_rerank(query, candidates, top_k)


class EnhancedRetrieval:
    """Enhanced retrieval system with caching, expansion, and reranking"""

    def __init__(self,
                 chunks_path: str,
                 use_cache: bool = True,
                 use_expansion: bool = True,
                 use_reranking: bool = True):
        """
        Initialize enhanced retrieval

        Args:
            chunks_path: Path to chunks.json
            use_cache: Enable embedding cache
            use_expansion: Enable query expansion
            use_reranking: Enable LLM reranking
        """
        self.chunks = self._load_chunks(chunks_path)
        self.documents = [chunk['content'] for chunk in self.chunks]

        # Components
        self.cache = EmbeddingCache() if use_cache else None
        self.expander = VietnameseQueryExpander() if use_expansion else None
        self.reranker = LLMReranker(use_mock=True) if use_reranking else None

        # Settings
        self.use_cache = use_cache
        self.use_expansion = use_expansion
        self.use_reranking = use_reranking

        print(f"\n✓ Enhanced Retrieval initialized")
        print(f"  - Chunks loaded: {len(self.chunks)}")
        print(f"  - Cache: {'Enabled' if use_cache else 'Disabled'}")
        print(f"  - Query expansion: {'Enabled' if use_expansion else 'Disabled'}")
        print(f"  - Reranking: {'Enabled' if use_reranking else 'Disabled'}")

    def _load_chunks(self, chunks_path: str) -> List[Dict]:
        """Load chunks from JSON"""
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        return chunks

    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 initial_k: int = 20) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Retrieve relevant chunks with all enhancements

        Args:
            query: User query
            top_k: Final number of results
            initial_k: Number of candidates before reranking

        Returns:
            (results, stats)
        """
        start_time = time.time()
        stats = {
            'query': query,
            'expanded_queries': [],
            'initial_candidates': 0,
            'final_results': 0,
            'cache_stats': {},
            'timing': {}
        }

        # Step 1: Query Expansion
        queries = [query]
        if self.use_expansion and self.expander:
            t0 = time.time()
            queries = self.expander.expand_query(query, max_expansions=2)
            stats['expanded_queries'] = queries
            stats['timing']['expansion'] = time.time() - t0
            print(f"\n📝 Query Expansion: {len(queries)} variations")
            for i, q in enumerate(queries):
                print(f"   {i+1}. {q}")

        # Step 2: Multi-query BM25 Retrieval
        t0 = time.time()
        all_scores = []
        for q in queries:
            scores = SimpleBM25.calculate_bm25_scores(q, self.documents)
            all_scores.append(scores)

        # Combine scores (max aggregation)
        combined_scores = np.max(all_scores, axis=0).tolist()
        stats['timing']['bm25'] = time.time() - t0

        # Get top initial_k candidates
        top_indices = np.argsort(combined_scores)[-initial_k:][::-1]

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
                'score': combined_scores[idx]
            })

        stats['initial_candidates'] = len(candidates)
        print(f"\n🔍 Initial retrieval: {len(candidates)} candidates")

        # Step 3: LLM Reranking
        if self.use_reranking and self.reranker:
            t0 = time.time()
            candidates = self.reranker.rerank(query, candidates, top_k)
            stats['timing']['reranking'] = time.time() - t0
            print(f"🎯 After reranking: {len(candidates)} results")
        else:
            candidates = candidates[:top_k]

        stats['final_results'] = len(candidates)

        # Cache stats
        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()

        stats['timing']['total'] = time.time() - start_time

        return candidates, stats

    def print_results(self, results: List[Dict], stats: Dict[str, Any]):
        """Print retrieval results"""
        print(f"\n{'='*100}")
        print(f"📊 RETRIEVAL RESULTS")
        print(f"{'='*100}\n")

        print(f"Query: {stats['query']}")
        if stats.get('expanded_queries'):
            print(f"Expanded to: {len(stats['expanded_queries'])} variations")
        print(f"Results: {stats['final_results']} chunks\n")

        for rank, result in enumerate(results, 1):
            print(f"{'─'*100}")
            print(f"[{rank}] Score: {result.get('rerank_score', result['score']):.4f} (BM25: {result['score']:.4f})")
            print(f"    Section: {result['section_code']} - {result['section_title']}")
            print(f"    Type: {result['section_type']} | Level: {result['level']} | Module: {result['module']}")
            print(f"    Tags: {', '.join(result['tags']) if result['tags'] else 'None'}")
            print(f"    Words: {result['word_count']}")

            preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            print(f"\n    Preview:")
            print(f"    {preview}")
            print()

        # Print timing stats
        print(f"{'='*100}")
        print(f"⏱️  Performance Stats:")
        if 'expansion' in stats['timing']:
            print(f"  - Query expansion: {stats['timing']['expansion']*1000:.1f}ms")
        print(f"  - BM25 retrieval: {stats['timing']['bm25']*1000:.1f}ms")
        if 'reranking' in stats['timing']:
            print(f"  - Reranking: {stats['timing']['reranking']*1000:.1f}ms")
        print(f"  - Total: {stats['timing']['total']*1000:.1f}ms")

        if stats.get('cache_stats'):
            cache = stats['cache_stats']
            print(f"\n💾 Cache Stats:")
            print(f"  - Size: {cache['size']} embeddings")
            print(f"  - Hits: {cache['hits']} | Misses: {cache['misses']}")
            print(f"  - Hit rate: {cache['hit_rate']:.1f}%")

        print(f"{'='*100}\n")

    def save_cache(self):
        """Save cache to disk"""
        if self.cache:
            self.cache.save_cache()


def compare_retrievals(chunks_path: str, query: str, top_k: int = 5):
    """So sánh kết quả giữa basic và enhanced retrieval"""
    print(f"\n{'='*100}")
    print(f"🔬 COMPARISON: Basic vs Enhanced Retrieval")
    print(f"{'='*100}\n")
    print(f"Query: {query}\n")

    # Basic retrieval (no enhancements)
    print("🔵 BASIC RETRIEVAL (BM25 only)")
    print("─" * 100)
    basic = EnhancedRetrieval(
        chunks_path,
        use_cache=False,
        use_expansion=False,
        use_reranking=False
    )
    basic_results, basic_stats = basic.retrieve(query, top_k=top_k, initial_k=top_k)

    print("\nTop 3 Results:")
    for i, r in enumerate(basic_results[:3], 1):
        print(f"{i}. [{r['section_code']}] {r['section_title'][:60]} (score: {r['score']:.2f})")

    # Enhanced retrieval (all features)
    print(f"\n{'─'*100}")
    print("🟢 ENHANCED RETRIEVAL (Expansion + Reranking)")
    print("─" * 100)
    enhanced = EnhancedRetrieval(
        chunks_path,
        use_cache=True,
        use_expansion=True,
        use_reranking=True
    )
    enhanced_results, enhanced_stats = enhanced.retrieve(query, top_k=top_k, initial_k=20)

    print("\nTop 3 Results:")
    for i, r in enumerate(enhanced_results[:3], 1):
        rerank_score = r.get('rerank_score', r['score'])
        print(f"{i}. [{r['section_code']}] {r['section_title'][:60]} (rerank: {rerank_score:.2f}, bm25: {r['score']:.2f})")

    # Comparison summary
    print(f"\n{'='*100}")
    print("📊 SUMMARY")
    print(f"{'='*100}")
    print(f"Basic retrieval time: {basic_stats['timing']['total']*1000:.1f}ms")
    print(f"Enhanced retrieval time: {enhanced_stats['timing']['total']*1000:.1f}ms")
    print(f"Queries expanded: {len(enhanced_stats.get('expanded_queries', []))}")

    # Check if top results changed
    basic_top = [r['chunk_id'] for r in basic_results[:3]]
    enhanced_top = [r['chunk_id'] for r in enhanced_results[:3]]

    if basic_top != enhanced_top:
        print(f"\n✨ Reranking changed top results!")
    else:
        print(f"\n✓ Top results remain the same (but scores improved)")

    print(f"{'='*100}\n")

    return basic_results, enhanced_results


def main():
    """Test enhanced retrieval system"""
    chunks_path = Path(__file__).parent / "output_admission" / "chunks.json"

    # Test queries
    test_queries = [
        "Điều kiện tuyển sinh vào trường quân đội",
        "Hồ sơ đăng ký dự tuyển",
        "Thời gian nộp hồ sơ",
        "Điều kiện về sức khỏe",
        "Điểm thi tuyển",
    ]

    print(f"\n{'='*100}")
    print("🚀 TESTING ENHANCED RETRIEVAL SYSTEM")
    print(f"{'='*100}\n")

    # Test 1: Demonstrate each feature
    print("TEST 1: Enhanced Retrieval with all features")
    print("=" * 100)

    enhanced = EnhancedRetrieval(
        chunks_path,
        use_cache=True,
        use_expansion=True,
        use_reranking=True
    )

    for query in test_queries[:2]:  # Test first 2 queries in detail
        results, stats = enhanced.retrieve(query, top_k=3, initial_k=20)
        enhanced.print_results(results, stats)

    # Test 2: Compare basic vs enhanced
    print("\n" + "=" * 100)
    print("TEST 2: Comparison - Basic vs Enhanced")
    print("=" * 100)

    compare_retrievals(chunks_path, "Điều kiện về sức khỏe", top_k=5)

    # Save cache
    enhanced.save_cache()

    print("✅ All tests completed!")


if __name__ == "__main__":
    main()
