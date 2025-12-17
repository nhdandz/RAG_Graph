"use client";

import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { HeroSection } from "@/components/layout/HeroSection";
import { Sidebar } from "@/components/layout/Sidebar";
import { ChatInterface } from "@/components/chat/ChatInterface";
import { InfoPanel } from "@/components/info/InfoPanel";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-bg-primary">
      {/* Header */}
      <Header />

      {/* Main Content */}
      <main className="flex-1">
        <div className="container mx-auto px-4 py-6">
          {/* Hero Section */}
          <HeroSection />

          {/* Main Content Grid - 3 columns */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Sidebar - Quick Links (Hidden on mobile, 2 cols on desktop) */}
            <aside className="hidden lg:block lg:col-span-2">
              <Sidebar />
            </aside>

            {/* Chat Interface (Full width on mobile, 7 cols on desktop) */}
            <section className="lg:col-span-7">
              <ChatInterface />
            </section>

            {/* Info Panel (Full width on mobile, 3 cols on desktop) */}
            <aside className="lg:col-span-3">
              <InfoPanel />
            </aside>
          </div>

          {/* Mobile Sidebar - Collapsible */}
          <div className="lg:hidden mt-6">
            <details className="group">
              <summary className="cursor-pointer list-none">
                <div className="flex items-center justify-between p-4 bg-white dark:bg-zinc-800 rounded-lg border-2 border-military-green-600 shadow-md">
                  <span className="font-semibold text-military-green-900 dark:text-military-green-400">
                    📋 Truy cập nhanh
                  </span>
                  <svg className="w-5 h-5 text-military-green-600 transition-transform group-open:rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </summary>
              <div className="mt-4">
                <Sidebar />
              </div>
            </details>
          </div>

          {/* Sections with IDs for navigation */}
          <div id="dieu-kien" className="mt-12 scroll-mt-20"></div>
          <div id="lo-trinh" className="mt-12 scroll-mt-20"></div>
          <div id="ho-so" className="mt-12 scroll-mt-20"></div>
          <div id="faq" className="mt-12 scroll-mt-20"></div>
          <div id="lien-he" className="mt-12 scroll-mt-20"></div>
        </div>
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
}
