"use client";

import React, { useState } from 'react';
import Link from 'next/link';

export const Header: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navLinks = [
    { href: '/', label: 'Trang chủ' },
    { href: '#dieu-kien', label: 'Điều kiện' },
    { href: '#lo-trinh', label: 'Lộ trình' },
    { href: '#faq', label: 'FAQ' },
    { href: '#lien-he', label: 'Liên hệ' }
  ];

  return (
    <header className="sticky top-0 z-50 military-gradient shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16 md:h-20">
          {/* Logo and Title */}
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-12 h-12 md:w-14 md:h-14 rounded-full flex items-center justify-center shadow-lg transition-transform group-hover:scale-105 overflow-hidden bg-white">
              <img src="/logos/Emblem_VPA.svg.png" alt="Logo Quân đội Nhân dân Việt Nam" className="w-full h-full object-contain p-1" />
            </div>
            <div className="hidden md:block">
              <h1 className="text-white font-bold text-lg md:text-xl leading-tight">
                QUÂN ĐỘI NHÂN DÂN VIỆT NAM
              </h1>
              <p className="text-gold-500 text-xs md:text-sm font-medium">
                Tư vấn Tuyển sinh 2025
              </p>
            </div>
            <div className="md:hidden">
              <h1 className="text-white font-bold text-base leading-tight">
                TUYỂN SINH QUÂN SỰ
              </h1>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center gap-1">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="px-4 py-2 text-white hover:text-gold-500 transition-colors font-medium text-sm"
              >
                {link.label}
              </a>
            ))}
            <Link
              href="/admin"
              className="ml-4 px-4 py-2 bg-gold-600 hover:bg-gold-500 text-military-green-900 font-semibold rounded-md transition-all text-sm shadow-md"
            >
              🔐 Admin
            </Link>
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="lg:hidden text-white p-2 hover:bg-military-green-900 rounded-md transition-colors"
            aria-label="Toggle menu"
          >
            {isMobileMenuOpen ? (
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            ) : (
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            )}
          </button>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="lg:hidden border-t border-military-green-900 py-4">
            <nav className="flex flex-col gap-2">
              {navLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  className="px-4 py-2 text-white hover:bg-military-green-900 rounded-md transition-colors font-medium"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  {link.label}
                </a>
              ))}
              <Link
                href="/admin"
                className="mx-4 mt-2 px-4 py-2 bg-gold-600 hover:bg-gold-500 text-military-green-900 font-semibold rounded-md transition-all text-center shadow-md"
                onClick={() => setIsMobileMenuOpen(false)}
              >
                🔐 Admin
              </Link>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};
