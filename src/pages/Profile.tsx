import React, { useState, useEffect } from "react";
import { Link, Navigate } from "react-router-dom";
import Navbar from "@/components/layout/Navbar";

interface Track {
  id: string;
  title: string;
  date: string;
  score: number;
  grade: string;
  audioSrc: string;
}

interface ArtistInfo {
  username: string;
  rank: number;
  avatarText: string;
  bio: string;
  avgScore: number;
  grade: string;
  skills: { name: string; rating: number }[];
}

export default function ProfilePage() {
  const [profile, setProfile] = useState<any>(null);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const token = localStorage.getItem("token");

  useEffect(() => {
    if (!token) return;

    const fetchProfile = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch("/api/artists/me", {
          headers: {
            "Authorization": `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error("Failed to load artist profile");
        }

        const data = await response.json();
        setProfile(data);

        // Map tracks
        const mappedTracks: Track[] = (data.tracks || []).map((t: any) => ({
          id: t.id.toString(),
          title: t.title,
          date: new Date(t.createdAt).toLocaleDateString(),
          score: t.totalScore || 0,
          grade: t.grade || "PENDING",
          audioSrc: t.audioUrl,
        }));
        setTracks(mappedTracks);

      } catch (err: any) {
        console.error("Profile fetch error:", err);
        setError(err.message || "Failed to load profile.");
      } finally {
        setLoading(false);
      }
    };

    fetchProfile();
  }, [token]);

  const handleDeleteTrack = async (trackId: string) => {
    if (!window.confirm("Are you sure you want to delete this track? This action cannot be undone.")) {
      return;
    }

    try {
      const response = await fetch(`/api/tracks/${trackId}`, {
        method: "DELETE",
        headers: {
          "Authorization": `Bearer ${token}`,
        },
      });

      if (response.ok) {
        setTracks((prev) => prev.filter((t) => t.id !== trackId));
        alert("Track deleted successfully.");
      } else {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || "Failed to delete track");
      }
    } catch (err: any) {
      console.error("Delete track error:", err);
      alert(err.message || "Failed to delete track.");
    }
  };

  // Route guard: Redirect if not logged in
  if (!token) {
    return <Navigate to="/login" replace />;
  }

  if (loading) {
    return (
      <div className="min-h-screen w-full flex flex-col bg-raprank-dark text-white relative">
        <Navbar />
        <div className="flex-grow flex items-center justify-center font-graffiti text-3xl text-raprank-neon">
          LOADING ARTIST PROFILE...
        </div>
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="min-h-screen w-full flex flex-col bg-raprank-dark text-white relative">
        <Navbar />
        <div className="flex-grow flex flex-col items-center justify-center p-6 text-center">
          <h2 className="font-graffiti text-4xl text-rose-500 mb-4">PROFILE ERROR</h2>
          <p className="font-sans text-lg text-raprank-skin/60 mb-6">{error || "Failed to load artist console."}</p>
          <Link to="/leaderboard" className="font-graffiti text-xl text-raprank-neon hover:underline">
            GO TO LEADERBOARD
          </Link>
        </div>
      </div>
    );
  }

  // Derive dynamic skills based on their average score
  const avg = Math.round(profile.avgScore || 0);
  const artistInfo: ArtistInfo = {
    username: profile.username,
    rank: profile.rank || 0,
    avatarText: profile.username.substring(0, 2).toUpperCase(),
    bio: profile.bio || "Step up to the mic and show the world your skills.",
    avgScore: avg,
    grade: profile.grade || "C",
    skills: [
      { name: "Rhyme Scheme Density", rating: Math.min(100, Math.round(avg * 1.06)) },
      { name: "Syllable Cadence", rating: Math.min(100, Math.round(avg * 0.98)) },
      { name: "Flow Stability", rating: Math.min(100, Math.round(avg * 1.02)) },
    ],
  };

  return (
    <div className="min-h-screen w-full flex flex-col bg-raprank-dark text-white relative">
      <Navbar />

      {/* Grid Pattern Backdrop */}
      <div className="absolute inset-0 bg-[radial-gradient(#2a1518_1.5px,transparent_1.5px)] [background-size:24px_24px] opacity-25 pointer-events-none z-0" />

      <main className="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 md:py-16 relative z-10">
        
        {/* Profile Split Layout Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* LEFT COLUMN: Artist Stats Console */}
          <section className="lg:col-span-4 bg-black/75 backdrop-blur-2xl rounded-3xl p-6 md:p-8 border-3 border-raprank-neon shadow-[0_0_30px_rgba(168,255,62,0.15)] flex flex-col items-center text-center" aria-label="Artist profile info">
            
            {/* Glowing Avatar */}
            <div className="relative h-28 w-28 flex items-center justify-center mb-4 select-none">
              <div className="absolute inset-0 rounded-full border-3 border-raprank-neon animate-pulse shadow-[0_0_12px_#a8ff3e]" />
              <div className="h-24 w-24 rounded-full bg-raprank-maroon border-2 border-raprank-neon/60 flex items-center justify-center font-graffiti text-3xl text-white shadow-inner">
                {artistInfo.avatarText}
              </div>
            </div>

            {/* Username & Badge */}
            <div className="mb-4">
              <h2 className="font-graffiti text-4xl text-white tracking-wide uppercase">
                {artistInfo.username}
              </h2>
              <span className="inline-block text-[10px] font-bold text-black bg-raprank-neon px-3 py-1 rounded-full uppercase tracking-widest mt-1">
                {profile.badgeTitle || "STREET SPITTER"}
              </span>
            </div>

            {/* Bio */}
            <p className="text-sm font-semibold text-raprank-skin/70 leading-relaxed border-t border-b border-raprank-maroon/20 py-4 mb-6">
              {artistInfo.bio}
            </p>

            {/* Quick Metrics Grid */}
            <div className="grid grid-cols-3 gap-2 w-full mb-6">
              <div className="bg-raprank-maroon/30 rounded-2xl py-3 border border-raprank-maroon/20">
                <span className="block text-[10px] font-bold text-raprank-skin/50 uppercase">RANK</span>
                <span className="font-graffiti text-xl text-raprank-neon">
                  {artistInfo.rank > 0 ? `#${artistInfo.rank}` : "-"}
                </span>
              </div>
              <div className="bg-raprank-maroon/30 rounded-2xl py-3 border border-raprank-maroon/20">
                <span className="block text-[10px] font-bold text-raprank-skin/50 uppercase">AVG</span>
                <span className="font-graffiti text-xl text-white">{artistInfo.avgScore}</span>
              </div>
              <div className="bg-raprank-maroon/30 rounded-2xl py-3 border border-raprank-maroon/20">
                <span className="block text-[10px] font-bold text-raprank-skin/50 uppercase">GRADE</span>
                <span className="font-graffiti text-xl text-raprank-neon">{artistInfo.grade}</span>
              </div>
            </div>

            {/* Skill Bars */}
            <div className="w-full space-y-3.5 text-left">
              <h3 className="text-xs font-bold text-raprank-neon uppercase tracking-widest border-b border-raprank-maroon/10 pb-1">
                Skill breakdown
              </h3>
              
              {artistInfo.skills.map((skill) => (
                <div key={skill.name} className="space-y-1">
                  <div className="flex justify-between text-xs font-bold text-white/80">
                    <span>{skill.name}</span>
                    <span className="text-raprank-neon">{skill.rating}%</span>
                  </div>
                  <div className="w-full bg-raprank-maroon/40 rounded-full h-2 overflow-hidden border border-raprank-maroon/25">
                    <div
                      className="bg-raprank-neon h-full rounded-full shadow-[0_0_8px_#a8ff3e]"
                      style={{ width: `${skill.rating}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* RIGHT COLUMN: Track Upload History */}
          <section className="lg:col-span-8 bg-black/70 backdrop-blur-xl border-3 border-raprank-maroon/40 rounded-3xl p-6 md:p-8 shadow-xl" aria-label="Track list upload history">
            <div className="flex justify-between items-center mb-6 pb-4 border-b border-raprank-maroon/20">
              <h2 className="font-graffiti text-3xl tracking-wider text-raprank-cream uppercase">
                RELEASE LISTING ({tracks.length})
              </h2>
              <span className="text-xs font-bold text-raprank-skin/50 uppercase tracking-widest">
                RAP SHEET
              </span>
            </div>

            {/* Tracks List */}
            <div className="space-y-6">
              {tracks.length === 0 ? (
                <div className="text-center py-12 text-sm font-semibold text-raprank-skin/40 uppercase tracking-widest">
                  No tracks dropped yet. Go to Upload to release your first bar!
                </div>
              ) : (
                tracks.map((track) => (
                  <div
                    key={track.id}
                    className="bg-raprank-maroon/25 hover:bg-raprank-maroon/35 transition-colors duration-150 border border-raprank-maroon/30 rounded-2xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4"
                  >
                    <div className="flex items-start space-x-4">
                      {/* Vinyl Record Icon */}
                      <div className="bg-raprank-neon/10 p-3 rounded-xl border border-raprank-neon/20 shrink-0 text-raprank-neon">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-7 w-7 animate-spin"
                          style={{ animationDuration: '8s' }}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={2}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
                          />
                        </svg>
                      </div>
                      
                      <div>
                        <Link to={`/tracks/${track.id}`}>
                          <h3 className="font-bold text-lg text-white hover:text-raprank-neon transition-colors duration-150 select-text cursor-pointer">
                            {track.title}
                          </h3>
                        </Link>
                        <p className="text-xs font-semibold text-raprank-skin/50 mt-0.5">
                          Released: {track.date}
                        </p>
                        
                        {/* Audio Preview controls */}
                        <div className="mt-3.5 max-w-[280px] md:max-w-xs">
                          <audio src={track.audioSrc} controls className="w-full h-8" />
                        </div>
                      </div>
                    </div>

                    {/* Score & Actions */}
                    <div className="flex md:flex-col items-center md:items-end justify-between border-t border-raprank-maroon/20 pt-4 md:pt-0 md:border-t-0 shrink-0 gap-4">
                      
                      <div className="flex items-center space-x-3 md:space-x-0 md:flex-col md:items-end">
                        <span className="font-sans text-xs font-bold text-raprank-skin/50 uppercase">SCORE</span>
                        <div className="flex items-baseline space-x-1 mt-0.5">
                          <span className="font-graffiti text-3xl text-raprank-neon">{track.score}</span>
                          <span className="font-graffiti text-lg text-raprank-neon">/100</span>
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center gap-2 md:justify-end">
                        <Link
                          to={`/tracks/${track.id}`}
                          className="text-xs font-bold tracking-widest uppercase text-white bg-transparent border border-raprank-cream/30 hover:border-raprank-neon hover:text-raprank-neon px-4 py-2.5 rounded-full transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-raprank-neon text-center"
                        >
                          VIEW BREAKDOWN
                        </Link>
                        <button
                          type="button"
                          onClick={() => handleDeleteTrack(track.id)}
                          className="text-xs font-bold tracking-widest uppercase text-rose-400 bg-red-950/20 border border-red-900/40 hover:bg-red-900/30 hover:border-rose-500 hover:text-rose-300 px-4 py-2.5 rounded-full transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 cursor-pointer"
                        >
                          DELETE
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </main>

      <footer className="w-full py-6 text-center text-xs text-white/20 font-bold tracking-widest z-10 border-t border-raprank-maroon/10">
        © 2026 RAPRANK. ARTIST MANAGEMENT CONSOLE.
      </footer>
    </div>
  );
}
