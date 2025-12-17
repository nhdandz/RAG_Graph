"use client";

import React, { useState, useEffect } from 'react';
import { LogoCarousel } from '../shared/LogoCarousel';
import { getLogos } from '@/lib/api/logos';
import type { Logo } from '@/lib/types/logo';

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();
  const [logos, setLogos] = useState<Logo[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Load logos from API
    const loadLogos = async () => {
      try {
        const data = await getLogos();
        setLogos(data);
      } catch (error) {
        console.error('Failed to load logos:', error);
        setLogos([]);
      } finally {
        setIsLoading(false);
      }
    };

    loadLogos();
  }, []);

  return (
    <footer className="mt-12 bg-military-green-900 text-white">
      {/* Logo Carousel Section */}
      <div className="border-t border-b border-gold-600 py-6 bg-military-green-dark">
        <div className="container mx-auto px-4">
          <p className="text-center text-gold-500 text-sm font-semibold mb-4">
            CÁC TRƯỜNG QUÂN SỰ THAM GIA TUYỂN SINH
          </p>
          {isLoading ? (
            <div className="flex justify-center items-center h-24">
              <div className="animate-spin w-8 h-8 border-4 border-gold-500 border-t-transparent rounded-full"></div>
            </div>
          ) : (
            <LogoCarousel logos={logos} autoplayInterval={5000} pauseOnHover={true} />
          )}
        </div>
      </div>

      {/* Contact Information */}
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Contact */}
          <div>
            <h3 className="text-gold-500 font-bold text-lg mb-3">LIÊN HỆ</h3>
            <div className="space-y-2 text-sm">
              <p className="flex items-center gap-2">
                <span className="text-gold-500">📞</span>
                <span>Hotline: 1900 xxxx</span>
              </p>
              <p className="flex items-center gap-2">
                <span className="text-gold-500">✉️</span>
                <span>Email: tuyensinh@mod.gov.vn</span>
              </p>
              <p className="flex items-center gap-2">
                <span className="text-gold-500">📍</span>
                <span>Hà Nội, Việt Nam</span>
              </p>
            </div>
          </div>

          {/* Office Hours */}
          <div>
            <h3 className="text-gold-500 font-bold text-lg mb-3">GIỜ LÀM VIỆC</h3>
            <div className="space-y-2 text-sm">
              <p>Thứ 2 - Thứ 6: 8:00 - 17:00</p>
              <p>Thứ 7: 8:00 - 12:00</p>
              <p className="text-gray-400">Chủ nhật: Nghỉ</p>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-gold-500 font-bold text-lg mb-3">LIÊN KẾT NHANH</h3>
            <div className="space-y-2 text-sm">
              <a href="#dieu-kien" className="block hover:text-gold-500 transition-colors">
                ▸ Điều kiện tuyển sinh
              </a>
              <a href="#lo-trinh" className="block hover:text-gold-500 transition-colors">
                ▸ Lộ trình đào tạo
              </a>
              <a href="#faq" className="block hover:text-gold-500 transition-colors">
                ▸ Câu hỏi thường gặp
              </a>
            </div>
          </div>
        </div>

        {/* Copyright */}
        <div className="mt-8 pt-6 border-t border-military-green-600 text-center text-sm text-gray-400">
          <p>
            © {currentYear} Quân đội Nhân dân Việt Nam. All rights reserved.
          </p>
          <p className="mt-1 text-xs">
            Được hỗ trợ bởi Hệ thống AI Tư vấn Tuyển sinh
          </p>
        </div>
      </div>
    </footer>
  );
};
