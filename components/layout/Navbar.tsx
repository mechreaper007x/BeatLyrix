"use client";

import React from "react";
import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const { pathname } = useLocation();

  const navLinks = [
    { name: "Leaderboard", href: "/leaderboard" },
    { name: "Upload", href: "/upload" },
    { name: "Profile", href: "/profile" },
  ];

  return (
    <header className="sticky top-0 z-50 w-full bg-raprank-dark/95 backdrop-blur-md border-b-2 border-raprank-maroon/30 shadow-[0_4px_20px_rgba(0,0,0,0.4)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
        
        {/* Logo and Tagline */}
        <div className="flex items-center space-x-8">
          <Link to="/upload" className="flex items-center space-x-2.5 group">
            <span className="font-graffiti text-3xl tracking-widest text-raprank-neon select-none drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)] transition-all duration-300 group-hover:scale-105">
              RAPRANK
            </span>
            <div className="h-2 w-2 rounded-full bg-raprank-neon shadow-[0_0_8px_#a8ff3e] animate-pulse" />
          </Link>
          
          {/* Desktop Nav Links */}
          <nav className="hidden md:flex items-center space-x-6">
            {navLinks.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  to={link.href}
                  className={`text-sm font-semibold tracking-wider uppercase transition-colors duration-200 px-3 py-1.5 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-raprank-neon ${
                    isActive
                      ? "text-raprank-neon bg-raprank-neon/10"
                      : "text-raprank-cream/70 hover:text-raprank-neon"
                  }`}
                >
                  {link.name}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Action Controls (Right side) */}
        <div className="flex items-center space-x-4">
          {/* Mobile view indicator */}
          <span className="md:hidden text-xs font-bold text-raprank-neon bg-raprank-neon/10 px-2.5 py-1 rounded-full uppercase">
            Upload
          </span>

          <Link
            to="/login"
            onClick={() => {
              localStorage.removeItem("token");
              localStorage.removeItem("username");
              localStorage.removeItem("artistId");
              localStorage.removeItem("badgeTitle");
            }}
            className="text-xs font-semibold tracking-widest uppercase text-rose-400 hover:text-rose-300 border border-rose-400/30 hover:border-rose-400/70 px-4 py-2 rounded-full transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 active:scale-[0.98]"
          >
            LOGOUT
          </Link>
        </div>
      </div>
    </header>
  );
}
