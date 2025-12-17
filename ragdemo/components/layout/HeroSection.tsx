"use client";

import React, { useState } from 'react';
import { Button } from '../shared/Button';

export const HeroSection: React.FC = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  if (isCollapsed) {
    return (
      <button
        onClick={() => setIsCollapsed(false)}
        className="w-full py-2 bg-military-green-100 dark:bg-military-green-900 text-military-green-900 dark:text-military-green-400 text-sm hover:bg-military-green-200 dark:hover:bg-military-green-800 transition-colors mb-6 rounded-lg"
      >
        ▼ Hiển thị banner
      </button>
    );
  }

  return (
    <div className="relative mb-8 bg-gradient-to-r from-military-green-900 via-military-green-600 to-military-green-900 rounded-xl shadow-2xl overflow-hidden">
      {/* Background Image */}
      <div className="absolute inset-0">
        <img
          src="/logos/qdnd.jpg"
          alt="Banner Quân đội Nhân dân Việt Nam"
          className="w-full h-full object-cover opacity-70"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-military-green-900/80 via-military-green-500/80 to-military-green-900/80" />
      </div>
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute inset-0" style={{
          backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(255,255,255,.1) 10px, rgba(255,255,255,.1) 20px)'
        }} />
      </div>

      <div className="relative px-6 py-12 md:py-16">
        <button
          onClick={() => setIsCollapsed(true)}
          className="absolute top-4 right-4 text-white/60 hover:text-white transition-colors"
          aria-label="Collapse banner"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
          </svg>
        </button>

        <div className="max-w-4xl mx-auto text-center">
          {/* Main Title */}
          <div className="mb-6">
            <div className="inline-block">
              <h1 className="text-3xl md:text-5xl font-bold text-white mb-2">
                TƯ VẤN TUYỂN SINH QUÂN SỰ 2025
              </h1>
              <div className="h-1 bg-gold-500 rounded-full"></div>
            </div>
          </div>

          {/* Subtitle */}
          <p className="text-lg md:text-xl text-gray-200 mb-8 max-w-2xl mx-auto">
            Hệ thống AI tư vấn tuyển sinh tự động - Hỗ trợ 24/7
            <br />
            <span className="text-gold-500 font-semibold">Nhanh chóng • Chính xác • Miễn phí</span>
          </p>

          {/* Quick Action Buttons */}
          <div className="flex flex-wrap justify-center gap-3 md:gap-4">
            <a href="#dieu-kien">
              <Button variant="primary" size="lg" className="bg-gold-600 hover:bg-gold-500 text-military-green-900">
                📋 Điều kiện tuyển sinh
              </Button>
            </a>
            <a href="#lo-trinh">
              <Button variant="outline" size="lg" className="border-white text-white hover:bg-white hover:text-military-green-900">
                🎯 Lộ trình đào tạo
              </Button>
            </a>
            <a href="#faq">
              <Button variant="outline" size="lg" className="border-white text-white hover:bg-white hover:text-military-green-900">
                ❓ Câu hỏi thường gặp
              </Button>
            </a>
          </div>

          {/* Stats */}
          <div className="mt-10 grid grid-cols-3 gap-4 md:gap-8 max-w-2xl mx-auto">
            <div className="text-center">
              <div className="text-2xl md:text-3xl font-bold text-gold-500">24/7</div>
              <div className="text-xs md:text-sm text-gray-300 mt-1">Hỗ trợ liên tục</div>
            </div>
            <div className="text-center border-l border-r border-white/30">
              <div className="text-2xl md:text-3xl font-bold text-gold-500">100%</div>
              <div className="text-xs md:text-sm text-gray-300 mt-1">Miễn phí</div>
            </div>
            <div className="text-center">
              <div className="text-2xl md:text-3xl font-bold text-gold-500">AI</div>
              <div className="text-xs md:text-sm text-gray-300 mt-1">Công nghệ hiện đại</div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Wave */}
      <div className="absolute bottom-0 left-0 right-0">
        <svg viewBox="0 0 1200 120" preserveAspectRatio="none" className="w-full h-8 md:h-12" style={{ transform: 'translateY(1px)' }}>
          <path d="M0,0 C150,80 350,0 600,40 C850,80 1050,0 1200,40 L1200,120 L0,120 Z" fill="currentColor" className="text-bg-primary" />
        </svg>
      </div>
    </div>
  );
};
