"use client";

import React, { useState } from 'react';
import { Card } from '../shared/Card';

// Placeholder InfoPanel - will be fully implemented in Phase 4
export const InfoPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'timeline' | 'criteria' | 'contact'>('timeline');

  const tabs = [
    { id: 'timeline' as const, label: 'Lịch trình', icon: '📅' },
    { id: 'criteria' as const, label: 'Điều kiện', icon: '📋' },
    { id: 'contact' as const, label: 'Liên hệ', icon: '📞' }
  ];

  return (
    <div className="space-y-4">
      {/* Tabs */}
      <div className="flex gap-1 bg-gray-100 dark:bg-zinc-800 p-1 rounded-lg">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 px-2 py-2 text-xs font-medium rounded-md transition-all ${
              activeTab === tab.id
                ? 'bg-military-green-600 text-white shadow-md'
                : 'text-gray-600 dark:text-gray-400 hover:text-military-green-600'
            }`}
          >
            <span className="mr-1">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <Card variant="military">
        {activeTab === 'timeline' && (
          <div>
            <h3 className="font-bold text-military-green-900 dark:text-military-green-400 mb-4 flex items-center gap-2">
              <span className="text-xl">📅</span>
              Lịch trình Tuyển sinh 2025
            </h3>
            <div className="space-y-3">
              {[
                { date: 'Tháng 3/2025', title: 'Mở đầu đăng ký', status: 'upcoming' },
                { date: 'Tháng 4/2025', title: 'Nộp hồ sơ', status: 'upcoming' },
                { date: 'Tháng 6/2025', title: 'Thi tuyển', status: 'upcoming' },
                { date: 'Tháng 7/2025', title: 'Công bố kết quả', status: 'upcoming' },
              ].map((event, idx) => (
                <div key={idx} className="flex gap-3">
                  <div className="flex-shrink-0 w-1 bg-military-green-600 rounded-full"></div>
                  <div className="flex-1">
                    <div className="text-sm font-semibold text-gray-900 dark:text-white">
                      {event.title}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {event.date}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'criteria' && (
          <div>
            <h3 className="font-bold text-military-green-900 dark:text-military-green-400 mb-4 flex items-center gap-2">
              <span className="text-xl">📋</span>
              Điều kiện Tuyển sinh
            </h3>
            <div className="space-y-3 text-sm">
              <div className="p-3 bg-green-50 dark:bg-zinc-900 rounded-md">
                <div className="font-semibold text-green-800 dark:text-green-400 mb-1">
                  ✓ Độ tuổi
                </div>
                <div className="text-gray-700 dark:text-gray-300 text-xs">
                  18-25 tuổi (nam), 18-23 tuổi (nữ)
                </div>
              </div>
              <div className="p-3 bg-blue-50 dark:bg-zinc-900 rounded-md">
                <div className="font-semibold text-blue-800 dark:text-blue-400 mb-1">
                  ✓ Học lực
                </div>
                <div className="text-gray-700 dark:text-gray-300 text-xs">
                  Tốt nghiệp THPT trở lên
                </div>
              </div>
              <div className="p-3 bg-purple-50 dark:bg-zinc-900 rounded-md">
                <div className="font-semibold text-purple-800 dark:text-purple-400 mb-1">
                  ✓ Sức khỏe
                </div>
                <div className="text-gray-700 dark:text-gray-300 text-xs">
                  Đạt tiêu chuẩn sức khỏe quân đội
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'contact' && (
          <div>
            <h3 className="font-bold text-military-green-900 dark:text-military-green-400 mb-4 flex items-center gap-2">
              <span className="text-xl">📞</span>
              Thông tin Liên hệ
            </h3>
            <div className="space-y-3 text-sm">
              <div className="flex items-start gap-2">
                <span className="text-gold-600 flex-shrink-0">📞</span>
                <div>
                  <div className="font-semibold text-gray-900 dark:text-white">Hotline</div>
                  <div className="text-gray-600 dark:text-gray-400 text-xs">1900 xxxx</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-gold-600 flex-shrink-0">✉️</span>
                <div>
                  <div className="font-semibold text-gray-900 dark:text-white">Email</div>
                  <div className="text-gray-600 dark:text-gray-400 text-xs">tuyensinh@mod.gov.vn</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-gold-600 flex-shrink-0">📍</span>
                <div>
                  <div className="font-semibold text-gray-900 dark:text-white">Địa chỉ</div>
                  <div className="text-gray-600 dark:text-gray-400 text-xs">Hà Nội, Việt Nam</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-gold-600 flex-shrink-0">🕐</span>
                <div>
                  <div className="font-semibold text-gray-900 dark:text-white">Giờ làm việc</div>
                  <div className="text-gray-600 dark:text-gray-400 text-xs">Thứ 2-6: 8:00-17:00</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};
