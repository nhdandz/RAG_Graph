"use client";

import React, { useState, useEffect } from 'react';
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

const LOGOS_PER_SLIDE = 5;

export const LogoCarousel: React.FC<LogoCarouselProps> = ({
  logos,
  autoplayInterval = 5000,
  pauseOnHover = true
}) => {
  const [isPaused, setIsPaused] = useState(false);
  const [currentSlide, setCurrentSlide] = useState(0);

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

  // Calculate total slides
  const totalSlides = Math.ceil(activeLogos.length / LOGOS_PER_SLIDE);

  // Group logos into slides
  const slides: Logo[][] = [];
  for (let i = 0; i < totalSlides; i++) {
    const start = i * LOGOS_PER_SLIDE;
    const end = start + LOGOS_PER_SLIDE;
    slides.push(activeLogos.slice(start, end));
  }

  // Navigation functions
  const goToSlide = (index: number) => {
    setCurrentSlide(index);
  };

  const goToNextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % totalSlides);
  };

  const goToPrevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + totalSlides) % totalSlides);
  };

  // Auto-play effect
  useEffect(() => {
    if (isPaused || totalSlides <= 1) return;

    const interval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % totalSlides);
    }, autoplayInterval);

    return () => clearInterval(interval);
  }, [isPaused, autoplayInterval, totalSlides]);

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
        {/* Slides Container */}
        <div className="overflow-hidden">
          <div
            className="logo-slides-container flex"
            style={{
              transform: `translateX(-${currentSlide * 100}%)`,
            }}
          >
            {slides.map((slideLogos, slideIndex) => (
              <div
                key={slideIndex}
                className="slide w-full flex-shrink-0 flex gap-8 items-center justify-center px-8"
              >
                {slideLogos.map((logo) => (
                  <div
                    key={logo.id}
                    className="logo-item flex flex-col items-center gap-3 min-w-[200px] transition-transform duration-300 hover:scale-110"
                    onClick={() => handleLogoClick(logo)}
                    style={{ cursor: logo.websiteUrl ? 'pointer' : 'default' }}
                  >
                    {/* Logo Image */}
                    <div className="relative w-32 h-20 flex items-center justify-center bg-white rounded-lg shadow-lg border-2 border-gold-500 p-3">
                      {logo.imageUrl ? (
                        <Image
                          src={logo.imageUrl}
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
            ))}
          </div>
        </div>

        {/* Navigation Buttons */}
        {totalSlides > 1 && (
          <>
            <button
              onClick={goToPrevSlide}
              className="absolute left-2 top-1/2 -translate-y-1/2 p-2 bg-military-green-900/80 hover:bg-military-green-900 rounded-full transition-all z-10 opacity-0 hover:opacity-100 group-hover:opacity-100"
              aria-label="Previous slide"
            >
              <svg className="w-4 h-4 text-gold-500" fill="currentColor" viewBox="0 0 24 24">
                <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z" />
              </svg>
            </button>
            <button
              onClick={goToNextSlide}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-military-green-900/80 hover:bg-military-green-900 rounded-full transition-all z-10 opacity-0 hover:opacity-100 group-hover:opacity-100"
              aria-label="Next slide"
            >
              <svg className="w-4 h-4 text-gold-500" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8.59 16.59L10 18l6-6-6-6-1.41 1.41L13.17 12z" />
              </svg>
            </button>
          </>
        )}

        {/* Pause/Play Button */}
        {totalSlides > 1 && (
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
        )}

        {/* Gradient Overlays */}
        <div className="absolute left-0 top-0 bottom-0 w-20 bg-gradient-to-r from-military-green-dark to-transparent pointer-events-none z-10" />
        <div className="absolute right-0 top-0 bottom-0 w-20 bg-gradient-to-l from-military-green-dark to-transparent pointer-events-none z-10" />
      </div>

      {/* Pagination Dots */}
      {totalSlides > 1 && (
        <div className="pagination-dots flex justify-center items-center gap-2 mt-3">
          {Array.from({ length: totalSlides }).map((_, index) => (
            <button
              key={index}
              onClick={() => goToSlide(index)}
              className={`pagination-dot w-2 h-2 rounded-full transition-all ${
                currentSlide === index
                  ? 'bg-gold-500 scale-125'
                  : 'bg-gray-400 hover:bg-gray-300'
              }`}
              aria-label={`Go to slide ${index + 1}`}
            />
          ))}
        </div>
      )}

      {/* Counter */}
      <div className="text-center mt-2 text-xs text-gray-400">
        {activeLogos.length} trường quân sự
      </div>
    </div>
  );
};
