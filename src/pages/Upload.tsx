import React from "react";
import Navbar from "@/components/layout/Navbar";
import UploadForm from "@/components/upload/UploadForm";

export default function UploadPage() {
  return (
    <div className="min-h-screen w-full flex flex-col bg-raprank-dark text-white relative">
      {/* Sticky Navigation Bar */}
      <Navbar />

      {/* Dotted grid pattern backdrop */}
      <div className="absolute inset-0 bg-[radial-gradient(#2a1518_1.5px,transparent_1.5px)] [background-size:24px_24px] opacity-25 pointer-events-none z-0" />

      {/* Main Content Area */}
      <main className="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 md:py-16 relative z-10 flex flex-col items-center">
        {/* Page Headings */}
        <div className="w-full max-w-2xl text-center md:text-left mb-8 md:mb-12">
          <h1 className="font-graffiti text-5xl md:text-6xl text-white tracking-widest uppercase drop-shadow-[0_4px_8px_rgba(0,0,0,0.8)]">
            DROP YOUR <span className="text-raprank-neon drop-shadow-[0_0_10px_rgba(168,255,62,0.35)]">TRACK</span>
          </h1>
          <p className="font-sans font-semibold text-lg text-raprank-skin/60 mt-2 select-text">
            Lyrics get scored. Bars get judged. <span className="text-rose-400">No mercy.</span>
          </p>
        </div>

        {/* Upload Form Console */}
        <div className="w-full flex justify-center">
          <UploadForm />
        </div>
      </main>

      {/* Page bottom decorative footer info */}
      <footer className="w-full py-6 text-center text-xs text-white/20 font-bold tracking-widest z-10 border-t border-raprank-maroon/10">
        © 2026 RAPRANK. BUILT BY SPITTERS FOR SPITTERS.
      </footer>
    </div>
  );
}
