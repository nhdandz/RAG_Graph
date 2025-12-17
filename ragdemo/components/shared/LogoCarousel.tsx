"use client";

import React, { useState } from 'react';
import Image from 'next/image';
import './LogoCarousel.css';

export interface Logo {
  id: string;
  name: string;
  imageUrl: string;
  websiteUrl?: string;
  order: number;
  active: boolean;
}

export interface LogoCarouselProps {
  logos: Logo[];
  autoplayInterval?: number;  // milliseconds, default 5000
  pauseOnHover?: boolean;     // default true
}

export const LogoCarousel: React.FC<LogoCarouselProps> = ({
  logos,
  pauseOnHover = true
}) => {
  const [isPaused, setIsPaused] = useState(false);

  // Filter active logos and sort by order
  const activeLogos = logos
    .filter(logo => logo.active)
    .sort((a, b) => a.order - b.order);

  if (activeLogos.length === 0) {
    return (
      <div className="flex justify-center items-center h-20 text-gray-400 text-sm italic">
        Chưa có logo trường quân sự
      </div>
    );
  }

  // Duplicate logos for seamless infinite scroll
  const duplicatedLogos = [...activeLogos, ...activeLogos, ...activeLogos];

  const handleLogoClick = (logo: Logo) => {
    if (logo.websiteUrl) {
      window.open(logo.websiteUrl, '_blank');
    }
  };

  return (
    <div className="logo-carousel-wrapper relative w-full overflow-hidden">
      {/* Carousel Container */}
      <div
        className="relative py-4"
        onMouseEnter={() => pauseOnHover && setIsPaused(true)}
        onMouseLeave={() => setIsPaused(false)}
      >
        {/* Scrolling Logos */}
        <div
          className={`logo-scroll-container flex gap-8 items-center ${
            isPaused ? 'paused' : ''
          }`}
        >
          {duplicatedLogos.map((logo, index) => (
            <div
              key={`${logo.id}-${index}`}
              className="logo-item flex flex-col items-center gap-3 min-w-[200px] transition-transform duration-300 hover:scale-110"
              onClick={() => handleLogoClick(logo)}
              style={{ cursor: logo.websiteUrl ? 'pointer' : 'default' }}
            >
              {/* Logo Image */}
              <div className="relative w-32 h-20 flex items-center justify-center bg-white rounded-lg shadow-lg border-2 border-gold-500 p-3">
                {logo.imageUrl ? (
                  <Image
                    src={`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'}${logo.imageUrl}`}
                    alt={logo.name}
                    fill
                    className="object-contain p-2"
                    unoptimized
                  />
                ) : (
                  <span className="text-4xl">🎖️</span>
                )}
              </div>

              {/* Logo Name */}
              <div className="text-center max-w-[200px]">
                <p className="text-xs font-semibold text-gold-500 line-clamp-2">
                  {logo.name}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Pause/Play Button */}
        <button
          onClick={() => setIsPaused(!isPaused)}
          className="absolute top-2 right-2 p-2 bg-military-green-900/80 hover:bg-military-green-900 rounded-full transition-all z-10"
          aria-label={isPaused ? 'Play' : 'Pause'}
        >
          {isPaused ? (
            <svg className="w-4 h-4 text-gold-500" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          ) : (
            <svg className="w-4 h-4 text-gold-500" fill="currentColor" viewBox="0 0 24 24">
              <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
            </svg>
          )}
        </button>

        {/* Gradient Overlays for smooth fade effect */}
        <div className="absolute left-0 top-0 bottom-0 w-20 bg-gradient-to-r from-military-green-dark to-transparent pointer-events-none z-10" />
        <div className="absolute right-0 top-0 bottom-0 w-20 bg-gradient-to-l from-military-green-dark to-transparent pointer-events-none z-10" />
      </div>

      {/* Counter */}
      <div className="text-center mt-2 text-xs text-gray-400">
        {activeLogos.length} trường quân sự
      </div>
    </div>
  );
};
