"use client";

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Card } from '../shared/Card';
import { Button } from '../shared/Button';

// Placeholder ChatInterface - will be fully implemented in Phase 3
export const ChatInterface: React.FC = () => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [loading, setLoading] = useState(false);

  const API_BASE_URL = "http://localhost:8080";

  const handleSend = async () => {
    if (!query.trim()) return;

    const userMessage = { role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    setQuery('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, topK: 3 })
      });

      const data = await response.json();

      if (response.ok) {
        const aiMessage = {
          role: 'assistant',
          content: data.answer.replace(/<think>[\s\S]*?<\/think>/gi, "").trim()
        };
        setMessages(prev => [...prev, aiMessage]);
      } else {
        setMessages(prev => [...prev, { role: 'assistant', content: `Lỗi: ${data.error}` }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Lỗi: ${error}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card variant="military" className="flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-200 dark:border-zinc-700">
        <div className="flex items-center gap-2">
          <img src="/logos/logo_bot.png" alt="AI Chatbot" className="w-8 h-8" />
          <h2 className="text-xl font-bold text-military-green-900 dark:text-military-green-400">
            Trợ lý AI Tuyển sinh
          </h2>
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setMessages([])}
          disabled={messages.length === 0}
        >
          🗑️ Xóa
        </Button>
      </div>

      {/* Messages Area */}
      <div className="overflow-y-auto mb-4 space-y-4 min-h-[500px] max-h-[700px]">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">💬</div>
            <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Chào mừng đến với Hệ thống Tư vấn Tuyển sinh
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
              Hãy đặt câu hỏi để bắt đầu
            </p>

            {/* Example Questions */}
            <div className="max-w-md mx-auto space-y-2">
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">Câu hỏi gợi ý:</p>
              {[
                "Điều kiện tuyển sinh là gì?",
                "Lộ trình đào tạo như thế nào?",
                "Thời gian nộp hồ sơ khi nào?"
              ].map((q, i) => (
                <button
                  key={i}
                  onClick={() => setQuery(q)}
                  className="block w-full px-4 py-2 text-sm text-left bg-gray-100 dark:bg-zinc-800 hover:bg-military-green-100 dark:hover:bg-military-green-900 rounded-md transition-colors"
                >
                  💡 {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex gap-3 max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                {/* Avatar */}
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                  msg.role === 'user'
                    ? 'bg-accent-blue text-white'
                    : 'bg-military-green-600 text-white'
                }`}>
                  {msg.role === 'user' ? '👤' : <img src="/logos/logo_bot.png" alt="AI Chatbot" className="w-8 h-8" />}
                </div>

                {/* Message Bubble */}
                <div className={`px-4 py-3 rounded-lg ${
                  msg.role === 'user'
                    ? 'bg-accent-blue text-white'
                    : 'bg-gray-100 dark:bg-zinc-800 text-gray-900 dark:text-gray-100 border-2 border-military-green-600'
                }`}>
                  {msg.role === 'user' ? (
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  ) : (
                    <div className="text-sm markdown-content">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))
        )}

        {loading && (
          <div className="flex justify-start">
            <div className="flex gap-3 max-w-[80%]">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-military-green-600 text-white flex items-center justify-center">
                <img src="/logos/icons8-chatbot-96.png" alt="AI" className="w-5 h-5" />
              </div>
              <div className="px-4 py-3 rounded-lg bg-gray-100 dark:bg-zinc-800 border-2 border-military-green-600">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-military-green-600 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-military-green-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                  <div className="w-2 h-2 bg-military-green-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="flex gap-2 -mx-6 -mb-6 p-4 bg-gray-50 dark:bg-zinc-900 border-t border-gray-200 dark:border-zinc-700 rounded-b-lg">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
          placeholder="Nhập câu hỏi của bạn..."
          className="flex-1 px-4 py-3 border-2 border-gray-300 dark:border-zinc-600 rounded-lg focus:outline-none focus:border-military-green-600 bg-white dark:bg-zinc-700 text-gray-900 dark:text-white"
          disabled={loading}
        />
        <Button
          onClick={handleSend}
          disabled={!query.trim() || loading}
          className="px-6"
        >
          {loading ? '⏳' : '📤'} Gửi
        </Button>
      </div>
    </Card>
  );
};
