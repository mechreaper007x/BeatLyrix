import React from "react";
import LoginSignupCard from "@/components/auth/LoginSignupCard";

export default function LoginPage() {
  return (
    <main 
      className="min-h-screen w-full flex items-center bg-[#1c1410] bg-[url(/images/background.jpg)] bg-cover bg-center bg-no-repeat relative overflow-y-auto"
      aria-label="RapRank talent platform authentication"
    >
      {/* Dark overlay for readability and premium contrast */}
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[0.5px] z-0" />

      {/* Decorative Brand Header (Desktop only) */}
      <div className="absolute top-8 left-8 z-20 hidden md:flex items-center space-x-3 select-none">
        <span className="font-graffiti text-4xl tracking-widest text-raprank-neon drop-shadow-[0_4px_6px_rgba(0,0,0,0.9)]">
          RAPRANK
        </span>
        <div className="h-2.5 w-2.5 rounded-full bg-raprank-neon animate-pulse shadow-[0_0_8px_#a8ff3e]" />
      </div>

      {/* Main Content Layout Container */}
      <div className="w-full max-w-7xl mx-auto px-6 md:px-12 lg:px-20 py-16 flex flex-col md:flex-row items-center md:items-center justify-between relative z-10">
        
        {/* Left Side: Tagline / Heading */}
        <div className="w-full md:w-1/2 flex flex-col items-center md:items-start text-center md:text-left mb-10 md:mb-0">
          <div className="md:hidden flex items-center space-x-2.5 mb-6">
            <h1 className="font-graffiti text-5xl tracking-widest text-raprank-neon drop-shadow-[0_3px_5px_rgba(0,0,0,0.9)]">
              RAPRANK
            </h1>
            <div className="h-2 w-2 rounded-full bg-raprank-neon animate-pulse" />
          </div>

          <div className="max-w-md hidden md:block">
            <h1 className="font-graffiti text-6xl lg:text-7xl leading-none text-white tracking-wide uppercase drop-shadow-[0_4px_8px_rgba(0,0,0,0.8)] mb-4">
              CHOOSE <br />
              <span className="text-raprank-neon drop-shadow-[0_0_10px_rgba(168,255,62,0.3)]">YOUR FLOW</span>
            </h1>
            <p className="font-sans font-semibold text-lg text-raprank-skin/80 leading-relaxed drop-shadow-[0_2px_4px_rgba(0,0,0,0.6)]">
              Battles. Ranks. Street Cred. <br />
              Step up to the mic and show the world your skills.
            </p>
          </div>
        </div>

        {/* Right Side: Renders the glass login/signup card */}
        <div className="w-full md:w-1/2 flex justify-center md:justify-end">
          <LoginSignupCard />
        </div>
      </div>

      {/* Subtle bottom credit */}
      <div className="absolute bottom-4 left-6 md:left-auto md:right-8 z-20 text-xs text-white/50 font-semibold tracking-widest select-none drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)]">
        © 2026 RAPRANK. STREET BATTLES PLATFORM.
      </div>
    </main>
  );
}
