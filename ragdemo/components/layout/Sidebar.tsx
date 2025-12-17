"use client";

import React from 'react';
import { Card } from '../shared/Card';

export const Sidebar: React.FC = () => {
  const quickLinks = [
    { href: '#dieu-kien', icon: '📋', label: 'Điều kiện', color: 'text-blue-600' },
    { href: '#lo-trinh', icon: '🎯', label: 'Lộ trình', color: 'text-green-600' },
    { href: '#ho-so', icon: '📁', label: 'Hồ sơ', color: 'text-purple-600' },
    { href: '#faq', icon: '❓', label: 'FAQ', color: 'text-orange-600' },
    { href: '#lien-he', icon: '📞', label: 'Liên hệ', color: 'text-red-600' }
  ];

  return (
    <div className="space-y-4">
      {/* Quick Links Card */}
      <Card variant="military" title="Truy cập nhanh">
        <nav className="space-y-1">
          {quickLinks.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="flex items-center gap-3 px-3 py-2.5 rounded-md hover:bg-military-green-100 dark:hover:bg-military-green-900 transition-colors group"
            >
              <span className="text-xl flex-shrink-0">{link.icon}</span>
              <span className="font-medium text-sm text-gray-700 dark:text-gray-300 group-hover:text-military-green-900 dark:group-hover:text-military-green-400">
                {link.label}
              </span>
            </a>
          ))}
        </nav>
      </Card>

      {/* Info Card */}
      <Card className="bg-gradient-to-br from-gold-50 to-amber-50 dark:from-zinc-800 dark:to-zinc-900 border-gold-600">
        <div className="text-center">
          <div className="text-3xl mb-2">🎖️</div>
          <h3 className="font-bold text-military-green-900 dark:text-gold-500 mb-2">
            Hotline Tư vấn
          </h3>
          <p className="text-2xl font-bold text-vietnam-red-600 mb-1">
            1900 xxxx
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            Hỗ trợ 24/7
          </p>
        </div>
      </Card>

      {/* Tips Card */}
      <Card className="bg-blue-50 dark:bg-zinc-800 border-blue-300">
        <div>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xl">💡</span>
            <h3 className="font-bold text-blue-900 dark:text-blue-400 text-sm">
              Mẹo sử dụng
            </h3>
          </div>
          <ul className="space-y-2 text-xs text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-2">
              <span className="text-blue-600 flex-shrink-0">▸</span>
              <span>Hỏi câu hỏi cụ thể để có câu trả lời chính xác</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 flex-shrink-0">▸</span>
              <span>Xem nguồn trích dẫn để kiểm chứng thông tin</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 flex-shrink-0">▸</span>
              <span>Liên hệ hotline nếu cần hỗ trợ thêm</span>
            </li>
          </ul>
        </div>
      </Card>
    </div>
  );
};
