"use client";

import { useState } from "react";

interface ContextChunk {
  type: "primary" | "secondary" | "parent_context" | "semantic_neighbor";
  heading: string;
  headingPath: string;
  content: string;
  similarity?: number;
  importance?: number;
  relatedTo?: string;
  relationshipType?: string; // parent, child, sibling, related
}

interface ContextStructure {
  chunks: ContextChunk[];
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [retrievedDocs, setRetrievedDocs] = useState<any[]>([]);
  const [contextStructure, setContextStructure] = useState<ContextStructure | null>(null);
  const [loading, setLoading] = useState(false);
  const [docCount, setDocCount] = useState(0);
  const [uploadStatus, setUploadStatus] = useState("");
  const [expandedDocs, setExpandedDocs] = useState<Set<number>>(new Set());

  const API_BASE_URL = "http://localhost:8080/api";

  // Function to remove <think> tags from answer
  const cleanAnswer = (text: string) => {
    return text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  };

  // Toggle expanded state for a document
  const toggleDocExpansion = (idx: number) => {
    const newExpanded = new Set(expandedDocs);
    if (newExpanded.has(idx)) {
      newExpanded.delete(idx);
    } else {
      newExpanded.add(idx);
    }
    setExpandedDocs(newExpanded);
  };

  // Get badge info for chunk type
  const getChunkTypeBadge = (type: string) => {
    switch (type) {
      case "primary":
        return {
          label: "Primary Match",
          bgColor: "bg-blue-100 dark:bg-blue-900",
          textColor: "text-blue-800 dark:text-blue-200",
          borderColor: "border-blue-300 dark:border-blue-700",
        };
      case "secondary":
        return {
          label: "Secondary Match",
          bgColor: "bg-cyan-100 dark:bg-cyan-900",
          textColor: "text-cyan-800 dark:text-cyan-200",
          borderColor: "border-cyan-300 dark:border-cyan-700",
        };
      case "parent_context":
        return {
          label: "Parent Context",
          bgColor: "bg-purple-100 dark:bg-purple-900",
          textColor: "text-purple-800 dark:text-purple-200",
          borderColor: "border-purple-300 dark:border-purple-700",
        };
      case "semantic_neighbor":
        return {
          label: "Related Content",
          bgColor: "bg-green-100 dark:bg-green-900",
          textColor: "text-green-800 dark:text-green-200",
          borderColor: "border-green-300 dark:border-green-700",
        };
      default:
        return {
          label: "Unknown",
          bgColor: "bg-gray-100 dark:bg-gray-900",
          textColor: "text-gray-800 dark:text-gray-200",
          borderColor: "border-gray-300 dark:border-gray-700",
        };
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUploadStatus("");
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setUploadStatus("Please select a file first");
      return;
    }

    setLoading(true);
    setUploadStatus("Uploading...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setUploadStatus(`Success! ${data.message}`);
        setDocCount(data.totalDocuments || 0);
        setFile(null);
      } else {
        setUploadStatus(`Error: ${data.error}`);
      }
    } catch (error) {
      setUploadStatus(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadFromJSON = async () => {
    setLoading(true);
    setUploadStatus("Loading chunks from JSON...");

    try {
      const response = await fetch(`${API_BASE_URL}/documents/load-from-json`, {
        method: "POST",
      });

      const data = await response.json();

      if (response.ok) {
        let statusMessage = `✅ Success! Loaded ${data.chunksAdded || data.totalDocuments} chunks từ Thông tư tuyển sinh.`;

        // Check if chunkStats exists (from enhanced.py)
        if (data.chunkStats && data.chunkStats.min !== undefined) {
          statusMessage += `\nChunk stats: ${data.chunkStats.min}-${data.chunkStats.max} từ (trung bình: ${Math.round(data.chunkStats.avg)})`;
        }

        setUploadStatus(statusMessage);
        setDocCount(data.totalDocuments || 0);
      } else {
        setUploadStatus(`❌ Error: ${data.detail || 'Failed to load chunks'}`);
      }
    } catch (error) {
      setUploadStatus(`❌ Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleQuery = async () => {
    if (!query.trim()) {
      return;
    }

    setLoading(true);
    setAnswer("");
    setRetrievedDocs([]);
    setContextStructure(null);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          topK: 3,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setAnswer(cleanAnswer(data.answer));
        setRetrievedDocs(data.retrievedDocuments || []);
        setContextStructure(data.contextStructure || null);
        setExpandedDocs(new Set()); // Reset expanded state
      } else {
        setAnswer(`Error: ${data.error}`);
      }
    } catch (error) {
      setAnswer(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-900 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-zinc-900 dark:text-white">
            RAG Demo 1
          </h1>
          <a
            href="/chunks-viewer"
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700
              transition-all font-semibold text-sm"
          >
            📊 View All Chunks
          </a>
        </div>

        {/* Document Upload Section */}
        <div className="bg-white dark:bg-zinc-800 rounded-lg p-6 mb-6 shadow">
          <h2 className="text-xl font-semibold mb-4 text-zinc-900 dark:text-white">
            Upload Document
          </h2>

          {/* Upload from file */}
          <div className="flex gap-4 items-start mb-4">
            <input
              type="file"
              onChange={handleFileChange}
              accept=".txt,.pdf,.docx"
              className="block w-full text-sm text-zinc-500 dark:text-zinc-400
                file:mr-4 file:py-2 file:px-4
                file:rounded-md file:border-0
                file:text-sm file:font-semibold
                file:bg-zinc-100 file:text-zinc-700
                dark:file:bg-zinc-700 dark:file:text-zinc-300
                hover:file:bg-zinc-200 dark:hover:file:bg-zinc-600"
            />
            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700
                disabled:bg-zinc-300 disabled:cursor-not-allowed whitespace-nowrap"
            >
              {loading ? "Uploading..." : "Upload"}
            </button>
          </div>

          {/* Divider */}
          <div className="relative my-4">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-zinc-300 dark:border-zinc-600"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white dark:bg-zinc-800 text-zinc-500 dark:text-zinc-400">
                OR
              </span>
            </div>
          </div>

          {/* Load from JSON */}
          <div className="flex items-center gap-4">
            <div className="flex-1 text-sm text-zinc-600 dark:text-zinc-400">
              <p className="font-medium mb-1">Load Admission Document Chunks</p>
              <p className="text-xs text-zinc-500 dark:text-zinc-500">
                Structured chunks từ Thông tư tuyển sinh với cấu trúc phân cấp (chunks.json)
              </p>
            </div>
            <button
              onClick={handleLoadFromJSON}
              disabled={loading}
              className="px-6 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700
                disabled:bg-zinc-300 disabled:cursor-not-allowed whitespace-nowrap
                flex items-center gap-2"
            >
              {loading ? "Loading..." : "📂 Load Admission Chunks"}
            </button>
          </div>

          {uploadStatus && (
            <p className="mt-4 text-sm text-zinc-600 dark:text-zinc-400 whitespace-pre-line">
              {uploadStatus}
            </p>
          )}
          <p className="mt-3 text-sm text-zinc-500 dark:text-zinc-400 font-medium">
            Total documents in store: {docCount}
          </p>
        </div>

        {/* Query Section */}
        <div className="bg-white dark:bg-zinc-800 rounded-lg p-6 mb-6 shadow">
          <h2 className="text-xl font-semibold mb-4 text-zinc-900 dark:text-white">
            Ask a Question
          </h2>
          <div className="flex gap-4">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleQuery()}
              placeholder="Enter your question..."
              className="flex-1 px-4 py-2 border border-zinc-300 dark:border-zinc-600
                rounded-md bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white
                focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={handleQuery}
              disabled={!query.trim() || loading}
              className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700
                disabled:bg-zinc-300 disabled:cursor-not-allowed"
            >
              {loading ? "Processing..." : "Ask"}
            </button>
          </div>
        </div>

        {/* Answer Section */}
        {answer && (
          <div className="bg-white dark:bg-zinc-800 rounded-lg p-6 mb-6 shadow">
            <h2 className="text-xl font-semibold mb-4 text-zinc-900 dark:text-white">
              Answer
            </h2>
            <p className="text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap">
              {answer}
            </p>
          </div>
        )}

        {/* Context Structure Section - Enhanced with all chunks sent to LLM */}
        {contextStructure && contextStructure.chunks && contextStructure.chunks.length > 0 ? (
          <div className="bg-white dark:bg-zinc-800 rounded-lg p-6 shadow">
            <h2 className="text-xl font-semibold mb-2 text-zinc-900 dark:text-white">
              Complete Context Sent to LLM
            </h2>
            <p className="text-sm text-zinc-600 dark:text-zinc-400 mb-4">
              This shows all chunks sent to the AI model, including primary matches, parent context, and related content
            </p>

            {/* Legend */}
            <div className="flex flex-wrap gap-3 mb-4 p-3 bg-zinc-50 dark:bg-zinc-900 rounded-md">
              <div className="flex items-center gap-2">
                <span className="px-2 py-1 text-xs rounded-md bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200">
                  Primary Match
                </span>
                <span className="text-xs text-zinc-600 dark:text-zinc-400">Direct query results</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-1 text-xs rounded-md bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200">
                  Parent Context
                </span>
                <span className="text-xs text-zinc-600 dark:text-zinc-400">Broader section context</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-1 text-xs rounded-md bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200">
                  Related Content
                </span>
                <span className="text-xs text-zinc-600 dark:text-zinc-400">Semantically similar</span>
              </div>
            </div>

            <div className="space-y-3">
              {contextStructure.chunks.map((chunk, idx) => {
                const isExpanded = expandedDocs.has(idx);
                const badge = getChunkTypeBadge(chunk.type);

                return (
                  <div
                    key={idx}
                    className={`border-2 ${badge.borderColor} rounded-md p-4 hover:shadow-md transition-all`}
                  >
                    <div
                      className="cursor-pointer"
                      onClick={() => toggleDocExpansion(idx)}
                    >
                      <div className="flex items-start gap-3 mb-2">
                        <span className="text-lg text-zinc-600 dark:text-zinc-400">
                          {isExpanded ? "▼" : "▶"}
                        </span>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 flex-wrap mb-2">
                            <span className={`px-2 py-1 text-xs rounded-md font-medium ${badge.bgColor} ${badge.textColor}`}>
                              {badge.label}
                            </span>
                            {chunk.relationshipType === 'child' && (
                              <span className="px-2 py-1 text-xs rounded-md font-medium bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200">
                                👶 Child Detail
                              </span>
                            )}
                            {chunk.relationshipType === 'sibling' && (
                              <span className="px-2 py-1 text-xs rounded-md font-medium bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200">
                                🤝 Sibling
                              </span>
                            )}
                            {chunk.similarity !== undefined && chunk.similarity !== null && chunk.similarity > 0 && (
                              <span className="text-xs text-zinc-500 dark:text-zinc-400 bg-zinc-100 dark:bg-zinc-700 px-2 py-1 rounded">
                                Similarity: {((chunk.similarity ?? 0) * 100).toFixed(2)}%
                              </span>
                            )}
                            {chunk.importance !== undefined && chunk.importance !== null && (
                              <span className="text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950 px-2 py-1 rounded">
                                Importance: {(chunk.importance * 100).toFixed(1)}%
                              </span>
                            )}
                          </div>
                          <div className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                            {chunk.heading}
                          </div>
                          <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                            {chunk.headingPath}
                          </div>
                          {chunk.relatedTo && (
                            <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-1 italic">
                              → {chunk.relationshipType === 'parent' ? 'Ngữ cảnh của' :
                                 chunk.relationshipType === 'child' ? 'Chi tiết của' :
                                 chunk.relationshipType === 'sibling' ? 'Cùng cấp với' :
                                 'Liên quan đến'}: {chunk.relatedTo}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                    {isExpanded && (
                      <div className="mt-3 pl-8 pr-2">
                        <div className="p-3 bg-zinc-50 dark:bg-zinc-900 rounded-md">
                          <p className="text-sm text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap leading-relaxed">
                            {chunk.content}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-950 rounded-md border border-blue-200 dark:border-blue-800">
              <p className="text-xs text-blue-800 dark:text-blue-200 mb-2">
                <strong>🎯 Total chunks sent to LLM: {contextStructure.chunks.length}</strong>
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-blue-800 dark:text-blue-200">
                <div className="flex items-center gap-1">
                  <span className="font-semibold">📌 Primary:</span>
                  <span>{contextStructure.chunks.filter(c => c.type === "primary").length}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="font-semibold">📂 Parents:</span>
                  <span>{contextStructure.chunks.filter(c => c.type === "parent_context").length}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="font-semibold">👶 Children:</span>
                  <span>{contextStructure.chunks.filter(c => c.relationshipType === "child").length}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="font-semibold">🤝 Siblings:</span>
                  <span>{contextStructure.chunks.filter(c => c.relationshipType === "sibling").length}</span>
                </div>
              </div>
            </div>
          </div>
        ) : retrievedDocs.length > 0 ? (
          // Fallback to simple view if contextStructure not available
          <div className="bg-white dark:bg-zinc-800 rounded-lg p-6 shadow">
            <h2 className="text-xl font-semibold mb-4 text-zinc-900 dark:text-white">
              Retrieved Context (Simple View)
            </h2>
            <div className="space-y-4">
              {retrievedDocs.map((doc, idx) => {
                const isExpanded = expandedDocs.has(idx);
                return (
                  <div
                    key={idx}
                    className="border border-zinc-200 dark:border-zinc-700 rounded-md p-4 hover:border-zinc-300 dark:hover:border-zinc-600 transition-colors"
                  >
                    <div
                      className="flex justify-between items-start mb-2 cursor-pointer"
                      onClick={() => toggleDocExpansion(idx)}
                    >
                      <div className="flex items-center gap-2 flex-1">
                        <span className="text-lg text-zinc-600 dark:text-zinc-400">
                          {isExpanded ? "▼" : "▶"}
                        </span>
                        <span className="text-sm font-medium text-zinc-600 dark:text-zinc-400">
                          {doc.filename}
                        </span>
                      </div>
                      <span className="text-sm text-zinc-500 dark:text-zinc-500">
                        Similarity: {(doc.similarity * 100).toFixed(2)}%
                      </span>
                    </div>
                    {isExpanded && (
                      <div className="mt-3 pl-7">
                        <p className="text-sm text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap">
                          {doc.content}
                        </p>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
