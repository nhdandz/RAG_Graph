"use client";

import { useState, useEffect } from "react";

interface ChunkInfo {
  index: number;
  chunk_id: string;
  heading: string;
  headingPath: string;
  level: number;
  section_code: string;
  section_type: string;
  module: string;
  wordCount: number;
  tags: string[];
  is_global_context: boolean;
  parent_id?: string;
  children_count: number;
  siblings_count: number;
  content?: string;
  contentPreview?: string;
  contentLength?: number;
}

interface ChunkStats {
  min: number;
  max: number;
  avg: number;
}

interface ChunkSummary {
  totalChunks: number;
  uniqueFiles: number;
  files: string[];
  chunkSizeStats: ChunkStats;
  typeDistribution: Record<string, number>;
  levelDistribution: Record<string, number>;
}

interface ChunksResponse {
  total: number;
  showing: number;
  offset: number;
  limit: number | null;
  chunks: ChunkInfo[];
  summary: ChunkSummary;
}

export default function ChunksViewer() {
  const [chunks, setChunks] = useState<ChunkInfo[]>([]);
  const [summary, setSummary] = useState<ChunkSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Filters
  const [selectedFile, setSelectedFile] = useState("");
  const [includeContent, setIncludeContent] = useState(false);
  const [limit, setLimit] = useState(20);
  const [offset, setOffset] = useState(0);

  // Pagination
  const [total, setTotal] = useState(0);

  // Expanded chunks
  const [expandedChunks, setExpandedChunks] = useState<Set<number>>(new Set());

  const API_BASE_URL = "http://localhost:8080/api";

  const loadChunks = async () => {
    setLoading(true);
    setError("");

    try {
      const params = new URLSearchParams({
        offset: offset.toString(),
        limit: limit.toString(),
        include_content: includeContent.toString(),
      });

      if (selectedFile) {
        params.append("filename", selectedFile);
      }

      const response = await fetch(
        `${API_BASE_URL}/documents/chunks?${params}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ChunksResponse = await response.json();

      setChunks(data.chunks);
      setSummary(data.summary);
      setTotal(data.total);
    } catch (err) {
      setError(`Error loading chunks: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadChunks();
  }, [offset, limit, selectedFile, includeContent]);

  const handlePrevPage = () => {
    if (offset > 0) {
      setOffset(Math.max(0, offset - limit));
      setExpandedChunks(new Set()); // Reset expanded state
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const handleNextPage = () => {
    if (offset + limit < total) {
      setOffset(offset + limit);
      setExpandedChunks(new Set()); // Reset expanded state
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const toggleChunkExpansion = (index: number) => {
    const newExpanded = new Set(expandedChunks);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedChunks(newExpanded);
  };

  const getTypeBadgeStyle = (type: string) => {
    switch (type) {
      case "root":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200";
      case "chuong":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
      case "muc":
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
      case "dieu":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
      case "khoan":
        return "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200";
      case "item_abc":
        return "bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200";
      case "item_dash":
        return "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200";
      case "item_plus":
        return "bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-100 via-blue-50 to-purple-100 dark:from-zinc-900 dark:via-zinc-800 dark:to-zinc-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white dark:bg-zinc-800 rounded-xl p-8 mb-6 shadow-lg">
          <div className="flex justify-between items-start mb-4">
            <div className="flex-1">
              <h1 className="text-4xl font-bold text-purple-600 dark:text-purple-400 mb-2">
                📊 Chunks Viewer
              </h1>
              <p className="text-zinc-600 dark:text-zinc-400">
                Xem và quản lý tất cả chunks trong hệ thống RAG
              </p>
            </div>
            <a
              href="/"
              className="px-4 py-2 bg-zinc-600 text-white rounded-lg hover:bg-zinc-700
                transition-all font-semibold text-sm whitespace-nowrap"
            >
              ← Back to Demo
            </a>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-white dark:bg-zinc-800 rounded-xl p-6 mb-6 shadow-lg">
          <div className="flex flex-wrap gap-4 items-center">
            <div className="flex items-center gap-2">
              <label className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                Filter by file:
              </label>
              <select
                value={selectedFile}
                onChange={(e) => {
                  setSelectedFile(e.target.value);
                  setOffset(0);
                }}
                className="px-3 py-2 border-2 border-zinc-200 dark:border-zinc-600 rounded-lg
                  bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white
                  focus:outline-none focus:border-purple-500"
              >
                <option value="">All files</option>
                {summary?.files.map((file) => (
                  <option key={file} value={file}>
                    {file}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2">
              <label className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                Limit:
              </label>
              <input
                type="number"
                value={limit}
                onChange={(e) => {
                  setLimit(parseInt(e.target.value) || 20);
                  setOffset(0);
                }}
                min="1"
                max="1000"
                className="w-20 px-3 py-2 border-2 border-zinc-200 dark:border-zinc-600 rounded-lg
                  bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white
                  focus:outline-none focus:border-purple-500"
              />
            </div>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={includeContent}
                onChange={(e) => setIncludeContent(e.target.checked)}
                className="w-4 h-4 text-purple-600 border-zinc-300 rounded
                  focus:ring-purple-500"
              />
              <span className="text-sm text-zinc-700 dark:text-zinc-300">
                Show full content
              </span>
            </label>

            <button
              onClick={() => {
                setOffset(0);
                loadChunks();
              }}
              className="px-5 py-2 bg-purple-600 text-white rounded-lg font-semibold
                hover:bg-purple-700 transition-all hover:shadow-lg"
            >
              🔄 Refresh
            </button>

            <button
              onClick={() => {
                if (expandedChunks.size === chunks.length) {
                  setExpandedChunks(new Set());
                } else {
                  setExpandedChunks(new Set(chunks.map(c => c.index)));
                }
              }}
              className="px-5 py-2 bg-blue-600 text-white rounded-lg font-semibold
                hover:bg-blue-700 transition-all hover:shadow-lg"
            >
              {expandedChunks.size === chunks.length ? "📕 Collapse All" : "📖 Expand All"}
            </button>
          </div>
        </div>

        {/* Summary Statistics */}
        {summary && (
          <div className="bg-white dark:bg-zinc-800 rounded-xl p-6 mb-6 shadow-lg">
            <h2 className="text-2xl font-semibold text-purple-600 dark:text-purple-400 mb-4">
              📈 Summary Statistics
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950 dark:to-purple-900 p-4 rounded-lg border-l-4 border-purple-500">
                <div className="text-xs text-purple-700 dark:text-purple-300 uppercase tracking-wide">
                  Total Chunks
                </div>
                <div className="text-3xl font-bold text-purple-700 dark:text-purple-300 mt-1">
                  {summary.totalChunks}
                </div>
              </div>

              <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950 dark:to-blue-900 p-4 rounded-lg border-l-4 border-blue-500">
                <div className="text-xs text-blue-700 dark:text-blue-300 uppercase tracking-wide">
                  Unique Files
                </div>
                <div className="text-3xl font-bold text-blue-700 dark:text-blue-300 mt-1">
                  {summary.uniqueFiles}
                </div>
              </div>

              <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-950 dark:to-green-900 p-4 rounded-lg border-l-4 border-green-500">
                <div className="text-xs text-green-700 dark:text-green-300 uppercase tracking-wide">
                  Min Size
                </div>
                <div className="text-3xl font-bold text-green-700 dark:text-green-300 mt-1">
                  {Math.round(summary.chunkSizeStats.min)}
                </div>
                <div className="text-xs text-green-600 dark:text-green-400">words</div>
              </div>

              <div className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-950 dark:to-orange-900 p-4 rounded-lg border-l-4 border-orange-500">
                <div className="text-xs text-orange-700 dark:text-orange-300 uppercase tracking-wide">
                  Max Size
                </div>
                <div className="text-3xl font-bold text-orange-700 dark:text-orange-300 mt-1">
                  {Math.round(summary.chunkSizeStats.max)}
                </div>
                <div className="text-xs text-orange-600 dark:text-orange-400">words</div>
              </div>

              <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 dark:from-indigo-950 dark:to-indigo-900 p-4 rounded-lg border-l-4 border-indigo-500">
                <div className="text-xs text-indigo-700 dark:text-indigo-300 uppercase tracking-wide">
                  Avg Size
                </div>
                <div className="text-3xl font-bold text-indigo-700 dark:text-indigo-300 mt-1">
                  {Math.round(summary.chunkSizeStats.avg)}
                </div>
                <div className="text-xs text-indigo-600 dark:text-indigo-400">words</div>
              </div>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="text-white text-xl">Loading chunks...</div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-500 text-white p-4 rounded-lg mb-6">
            {error}
          </div>
        )}

        {/* Chunks Grid */}
        {!loading && chunks.length === 0 ? (
          <div className="text-center py-12 text-zinc-600 dark:text-zinc-400">
            No chunks found
          </div>
        ) : (
          <div className="space-y-4">
            {chunks.map((chunk) => {
              const isExpanded = expandedChunks.has(chunk.index);
              return (
                <div
                  key={chunk.index}
                  className="bg-white dark:bg-zinc-800 rounded-xl p-5 shadow-lg
                    hover:shadow-xl transition-all duration-300 border-l-4 border-purple-500"
                >
                  {/* Header - Clickable */}
                  <div
                    className="flex justify-between items-start gap-4 mb-3 cursor-pointer"
                    onClick={() => toggleChunkExpansion(chunk.index)}
                  >
                    <div className="flex items-start gap-3 flex-1">
                      <span className="text-2xl text-zinc-500 dark:text-zinc-400 mt-0.5">
                        {isExpanded ? "▼" : "▶"}
                      </span>
                      <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
                        {chunk.heading}
                      </h3>
                    </div>
                    <div className="flex gap-2 flex-wrap justify-end">
                      <span
                        className={`px-3 py-1 text-xs font-semibold rounded-full uppercase tracking-wide ${getTypeBadgeStyle(
                          chunk.section_type
                        )}`}
                      >
                        {chunk.section_type}
                      </span>
                      <span className="px-3 py-1 text-xs font-semibold rounded-full bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                        L{chunk.level}
                      </span>
                      {chunk.section_code && (
                        <span className="px-3 py-1 text-xs font-semibold rounded-full bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200">
                          {chunk.section_code}
                        </span>
                      )}
                      {chunk.is_global_context && (
                        <span className="px-3 py-1 text-xs font-semibold rounded-full bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200">
                          🌍 Global Context
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Path */}
                  <div className="text-sm text-zinc-600 dark:text-zinc-400 italic mb-3 pl-10">
                    📍 {chunk.headingPath}
                  </div>

                  {/* Content - Collapsible */}
                  {isExpanded && (
                    <div className="pl-10 pr-2 animate-in fade-in duration-300">
                      <div className="bg-zinc-50 dark:bg-zinc-900 rounded-lg p-4 font-mono text-sm max-h-[600px] overflow-y-auto border border-zinc-200 dark:border-zinc-700">
                        <pre className="whitespace-pre-wrap text-zinc-700 dark:text-zinc-300 leading-relaxed">
                          {includeContent ? chunk.content : chunk.contentPreview}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* Meta */}
                  <div className="flex flex-wrap gap-4 mt-4 pt-4 border-t border-zinc-200 dark:border-zinc-700 text-sm text-zinc-600 dark:text-zinc-400 pl-10">
                    <span>
                      <strong className="text-zinc-900 dark:text-white">📚 {chunk.module}</strong>
                    </span>
                    <span>📊 Index: {chunk.index}</span>
                    <span>📝 {chunk.wordCount} words</span>
                    {!includeContent && chunk.contentLength && (
                      <span>📏 {chunk.contentLength} characters</span>
                    )}
                    {chunk.children_count > 0 && (
                      <span>👶 {chunk.children_count} children</span>
                    )}
                    {chunk.siblings_count > 0 && (
                      <span>🤝 {chunk.siblings_count} siblings</span>
                    )}
                    {chunk.tags.length > 0 && (
                      <span>🏷️ {chunk.tags.join(', ')}</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Pagination */}
        {!loading && chunks.length > 0 && (
          <div className="bg-white dark:bg-zinc-800 rounded-xl p-6 mt-6 shadow-lg">
            <div className="flex justify-between items-center">
              <div className="text-zinc-600 dark:text-zinc-400">
                Showing {offset + 1}-{Math.min(offset + chunks.length, total)} of{" "}
                {total} chunks
              </div>
              <div className="flex gap-3">
                <button
                  onClick={handlePrevPage}
                  disabled={offset === 0}
                  className="px-5 py-2 bg-purple-600 text-white rounded-lg font-semibold
                    hover:bg-purple-700 disabled:bg-zinc-300 disabled:cursor-not-allowed
                    transition-all"
                >
                  ← Previous
                </button>
                <button
                  onClick={handleNextPage}
                  disabled={offset + chunks.length >= total}
                  className="px-5 py-2 bg-purple-600 text-white rounded-lg font-semibold
                    hover:bg-purple-700 disabled:bg-zinc-300 disabled:cursor-not-allowed
                    transition-all"
                >
                  Next →
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
