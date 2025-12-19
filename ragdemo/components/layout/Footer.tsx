"use client";

import React from 'react';
import { LogoCarousel } from '../shared/LogoCarousel';
import type { Logo } from '@/lib/types/logo';

// Danh sách logo tĩnh từ thư mục public/logos
const STATIC_LOGOS: Logo[] = [
  {
    id: '1',
    name: 'Quân đội Nhân dân Việt Nam',
    imageUrl: '/logos/qdnd.jpg',
    websiteUrl: '#',
    order: 1,
    active: true
  },
  {
    id: '2',
    name: 'Học viện Quân y',
    imageUrl: '/logos/hocvienquany.webp',
    websiteUrl: '#',
    order: 2,
    active: true
  },
  {
    id: '3',
    name: 'Học viện Hải quân',
    imageUrl: '/logos/hocvienhaiquan.jpg',
    websiteUrl: '#',
    order: 3,
    active: true
  },
  {
    id: '4',
    name: 'Học viện Biên phòng',
    imageUrl: '/logos/hocvienbienphong.jpg',
    websiteUrl: '#',
    order: 4,
    active: true
  },
  {
    id: '5',
    name: 'Học viện Không quân',
    imageUrl: '/logos/hocvienkhqs.jpg',
    websiteUrl: '#',
    order: 5,
    active: true
  },
  {
    id: '6',
    name: 'Học viện Phòng không - Không quân',
    imageUrl: '/logos/hocvienpkkq.webp',
    websiteUrl: '#',
    order: 6,
    active: true
  },
  {
    id: '7',
    name: 'Học viện Hậu cần',
    imageUrl: '/logos/hvhk.jpg',
    websiteUrl: '#',
    order: 7,
    active: true
  },
  {
    id: '8',
    name: 'Học viện Lục quân',
    imageUrl: '/logos/hvlq.jpg',
    websiteUrl: '#',
    order: 8,
    active: true
  },
  {
    id: '9',
    name: 'Học viện Kỹ thuật Quân sự',
    imageUrl: '/logos/mta.png',
    websiteUrl: '#',
    order: 9,
    active: true
  },
  {
    id: '10',
    name: 'Trường Sĩ quan Chính trị',
    imageUrl: '/logos/truongsiquanchinhtri.jpg',
    websiteUrl: '#',
    order: 10,
    active: true
  },
  {
    id: '11',
    name: 'Trường Sĩ quan Công binh',
    imageUrl: '/logos/truongsiquancongbinh.webp',
    websiteUrl: '#',
    order: 11,
    active: true
  },
  {
    id: '12',
    name: 'Trường Sĩ quan Đặc công',
    imageUrl: '/logos/truongsiquandaccong.png',
    websiteUrl: '#',
    order: 12,
    active: true
  },
  {
    id: '13',
    name: 'Trường Sĩ quan Không quân',
    imageUrl: '/logos/truongsiquankhongquan.png',
    websiteUrl: '#',
    order: 13,
    active: true
  },
  {
    id: '14',
    name: 'Trường Sĩ quan Pháo binh',
    imageUrl: '/logos/truongsiquanphaobinh.jpg',
    websiteUrl: '#',
    order: 14,
    active: true
  },
  {
    id: '15',
    name: 'Trường Sĩ quan Phòng hóa',
    imageUrl: '/logos/truongsiquanphonghoa.png',
    websiteUrl: '#',
    order: 15,
    active: true
  },
  {
    id: '16',
    name: 'Trường Sĩ quan Tăng - Thiết giáp',
    imageUrl: '/logos/truongsiquantangthietgiap.png',
    websiteUrl: '#',
    order: 16,
    active: true
  },
  {
    id: '17',
    name: 'Trường Sĩ quan Thông tin',
    imageUrl: '/logos/truongsiquanthongtin.jpg',
    websiteUrl: '#',
    order: 17,
    active: true
  }
];

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="mt-12 bg-military-green-900 text-white">
      {/* Logo Carousel Section */}
      <div className="border-t border-b border-gold-600 py-6 bg-military-green-dark">
        <div className="container mx-auto px-4">
          <p className="text-center text-gold-500 text-sm font-semibold mb-4">
            CÁC TRƯỜNG QUÂN SỰ THAM GIA TUYỂN SINH
          </p>
          <LogoCarousel logos={STATIC_LOGOS} autoplayInterval={5000} pauseOnHover={true} />
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
