import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import Navbar from "@/components/layout/Navbar";
import { getApiUrl } from "@/src/utils/api";

interface Spitter {
  rank: number;
  artistId: number;
  username: string;
  avatarText: string;
  totalTracks: number;
  avgScore: number;
  grade: string;
  isCurrentUser: boolean;
}

export default function LeaderboardPage() {
  const [spitters, setSpitters] = useState<Spitter[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchLeaderboard = async () => {
      setLoading(true);
      setError(null);
      try {
        const token = localStorage.getItem("token");
        const headers: HeadersInit = token ? { "Authorization": `Bearer ${token}` } : {};

        const response = await fetch(getApiUrl("/api/leaderboard"), { headers });
        if (!response.ok) {
          throw new Error("Failed to load leaderboard database.");
        }

        const data = await response.json();
        const mappedSpitters: Spitter[] = data.map((item: any) => ({
          rank: item.rank,
          artistId: item.artistId,
          username: item.artistUsername,
          avatarText: item.artistUsername.substring(0, 2).toUpperCase(),
          totalTracks: item.trackCount,
          avgScore: Math.round(item.avgScore),
          grade: item.grade || "C",
          isCurrentUser: item.isCurrentUser,
        }));

        setSpitters(mappedSpitters);
      } catch (err: any) {
        console.error("Leaderboard fetch error:", err);
        setError(err.message || "Something went wrong.");
      } finally {
        setLoading(false);
      }
    };

    fetchLeaderboard();
  }, []);

  const filteredSpitters = spitters.filter((spitter) =>
    spitter.username.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Fallback mocks if database is completely empty
  const activeSpitters = filteredSpitters.length > 0 ? filteredSpitters : [];

  const topThree = spitters.slice(0, 3);
  const remainingSpitters = filteredSpitters.filter((s) => s.rank > 3);

  const getBadgeDetails = (rank: number) => {
    switch (rank) {
      case 1:
        return { color: "text-[#ffd700]", bg: "bg-[#ffd700]/10", border: "border-[#ffd700]", label: "👑 1ST" };
      case 2:
        return { color: "text-[#c0c0c0]", bg: "bg-[#c0c0c0]/10", border: "border-[#c0c0c0]", label: "⭐ 2ND" };
      case 3:
        return { color: "text-[#cd7f32]", bg: "bg-[#cd7f32]/10", border: "border-[#cd7f32]", label: "⚡ 3RD" };
      default:
        return { color: "text-raprank-neon", bg: "bg-raprank-neon/10", border: "border-raprank-neon", label: `#${rank}` };
    }
  };

  return (
    <div className="min-h-screen w-full flex flex-col bg-raprank-dark text-white relative">
      <Navbar />

      {/* Grid Pattern Backdrop */}
      <div className="absolute inset-0 bg-[radial-gradient(#2a1518_1.5px,transparent_1.5px)] [background-size:24px_24px] opacity-25 pointer-events-none z-0" />

      <main className="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 relative z-10">
        
        {/* Header Section */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-10 gap-6">
          <div className="text-center md:text-left">
            <h1 className="font-graffiti text-5xl md:text-6xl text-white tracking-widest uppercase drop-shadow-[0_4px_8px_rgba(0,0,0,0.8)]">
              TOP <span className="text-raprank-neon drop-shadow-[0_0_10px_rgba(168,255,62,0.35)]">SPITTERS</span>
            </h1>
            <p className="font-sans font-semibold text-lg text-raprank-skin/60 mt-2 select-text">
              The rankings don't lie. Spit fire or get left behind.
            </p>
          </div>

          {/* Search bar */}
          <div className="w-full md:max-w-xs relative">
            <label htmlFor="spitter-search" className="sr-only">
              Search Spitters
            </label>
            <input
              type="text"
              id="spitter-search"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="SEARCH SPITTERS..."
              className="w-full bg-raprank-maroon/60 text-white placeholder-raprank-skin/40 font-bold tracking-wider px-6 py-3.5 rounded-full border border-raprank-maroon/30 outline-none focus-visible:border-raprank-neon focus-visible:ring-4 focus-visible:ring-raprank-neon/30 transition-all duration-200"
            />
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 text-raprank-skin/40 absolute right-5 top-4.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2.5}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
        </div>

        {loading ? (
          <div className="text-center py-20 font-graffiti text-3xl tracking-widest text-[#a8ff3e]">
            LOADING RANKINGS DATABASE...
          </div>
        ) : error ? (
          <div className="text-center py-20">
            <p className="font-graffiti text-3xl text-rose-500 mb-2">ERROR CONNECTING BACKEND</p>
            <p className="font-sans font-semibold text-raprank-skin/60">{error}</p>
          </div>
        ) : spitters.length === 0 ? (
          <div className="text-center py-20">
            <p className="font-graffiti text-3xl text-white mb-2">THE CYPHER IS EMPTY</p>
            <p className="font-sans font-semibold text-raprank-skin/60">Upload your first track to claim the crown!</p>
          </div>
        ) : (
          <>
            {/* TOP 3 PODIUM SECTION */}
            {searchQuery === "" && topThree.length > 0 && (
              <section className="grid grid-cols-1 md:grid-cols-3 gap-6 items-end mb-12" aria-label="Podium rankings">
                
                {/* 2nd Place (Silver) */}
                {topThree[1] && (
                  <div className="order-2 md:order-1 bg-black/40 backdrop-blur-md rounded-3xl p-6 border-2 border-[#c0c0c0]/30 shadow-lg text-center flex flex-col items-center h-[260px] justify-center relative">
                    <div className="absolute -top-5 bg-[#c0c0c0] text-black px-4 py-1 rounded-full font-graffiti text-lg tracking-wider border-2 border-raprank-dark">
                      {getBadgeDetails(2).label}
                    </div>
                    <div className="h-16 w-16 rounded-full bg-raprank-maroon border-3 border-[#c0c0c0] flex items-center justify-center font-graffiti text-2xl text-white mb-3">
                      {topThree[1].avatarText}
                    </div>
                    <h3 className="font-graffiti text-2xl tracking-wide text-white">{topThree[1].username}</h3>
                    <p className="text-xs font-bold text-raprank-skin/50 mt-1 uppercase">Tracks: {topThree[1].totalTracks}</p>
                    <div className="mt-4 flex items-baseline space-x-1.5">
                      <span className="font-graffiti text-4xl text-[#c0c0c0]">{topThree[1].avgScore}</span>
                      <span className="text-xs font-bold text-[#c0c0c0]/70 uppercase">AVG</span>
                    </div>
                  </div>
                )}

                {/* 1st Place (Gold) */}
                {topThree[0] && (
                  <div className="order-1 md:order-2 bg-black/65 backdrop-blur-md rounded-3xl p-8 border-3 border-[#ffd700] shadow-2xl text-center flex flex-col items-center h-[310px] justify-center relative shadow-[#ffd700]/10">
                    <div className="absolute -top-6 bg-[#ffd700] text-black px-6 py-1.5 rounded-full font-graffiti text-xl tracking-wider border-2 border-raprank-dark shadow-[0_0_12px_#ffd700]">
                      {getBadgeDetails(1).label}
                    </div>
                    <div className="h-20 w-20 rounded-full bg-raprank-maroon border-4 border-[#ffd700] flex items-center justify-center font-graffiti text-3xl text-white mb-3 shadow-[0_0_15px_rgba(255,215,0,0.2)]">
                      {topThree[0].avatarText}
                    </div>
                    <h3 className="font-graffiti text-3xl tracking-wide text-white">{topThree[0].username}</h3>
                    <p className="text-sm font-bold text-raprank-skin/60 mt-1 uppercase">Tracks: {topThree[0].totalTracks}</p>
                    <div className="mt-4 flex items-baseline space-x-1.5">
                      <span className="font-graffiti text-5xl text-[#ffd700] drop-shadow-[0_0_8px_rgba(255,215,0,0.3)]">{topThree[0].avgScore}</span>
                      <span className="text-xs font-bold text-[#ffd700]/80 uppercase">AVG</span>
                    </div>
                  </div>
                )}

                {/* 3rd Place (Bronze) */}
                {topThree[2] && (
                  <div className="order-3 bg-black/40 backdrop-blur-md rounded-3xl p-6 border-2 border-[#cd7f32]/30 shadow-lg text-center flex flex-col items-center h-[230px] justify-center relative">
                    <div className="absolute -top-5 bg-[#cd7f32] text-black px-4 py-1 rounded-full font-graffiti text-lg tracking-wider border-2 border-raprank-dark">
                      {getBadgeDetails(3).label}
                    </div>
                    <div className="h-14 w-14 rounded-full bg-raprank-maroon border-3 border-[#cd7f32] flex items-center justify-center font-graffiti text-xl text-white mb-3">
                      {topThree[2].avatarText}
                    </div>
                    <h3 className="font-graffiti text-2xl tracking-wide text-white">{topThree[2].username}</h3>
                    <p className="text-xs font-bold text-raprank-skin/50 mt-1 uppercase">Tracks: {topThree[2].totalTracks}</p>
                    <div className="mt-3 flex items-baseline space-x-1.5">
                      <span className="font-graffiti text-3xl text-[#cd7f32]">{topThree[2].avgScore}</span>
                      <span className="text-xs font-bold text-[#cd7f32]/70 uppercase">AVG</span>
                    </div>
                  </div>
                )}

              </section>
            )}

            {/* REMAINING RANKINGS LIST CONTAINER */}
            <section className="bg-black/70 backdrop-blur-xl border-3 border-raprank-maroon/40 rounded-3xl p-6 shadow-xl" aria-label="Rankings list">
              <h2 className="font-graffiti text-3xl tracking-wide uppercase text-raprank-cream mb-6">
                {searchQuery === "" ? "RANKINGS #4 - #9" : `SEARCH RESULTS (${filteredSpitters.length})`}
              </h2>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-raprank-maroon/20 text-xs font-bold tracking-widest uppercase text-raprank-skin/60">
                      <th className="pb-4 pl-4 w-20">RANK</th>
                      <th className="pb-4">SPITTER</th>
                      <th className="pb-4 w-32">TRACKS</th>
                      <th className="pb-4 w-32">AVG LQI SCORE</th>
                      <th className="pb-4 w-24">GRADE</th>
                      <th className="pb-4 pr-4 text-right w-36">ACTION</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-raprank-maroon/10">
                    {(searchQuery === "" ? remainingSpitters : filteredSpitters).map((spitter) => {
                      const badge = getBadgeDetails(spitter.rank);

                      return (
                        <tr
                          key={spitter.username}
                          className={`group transition-colors duration-150 hover:bg-raprank-maroon/10 ${
                            spitter.isCurrentUser ? "bg-raprank-neon/5" : ""
                          }`}
                        >
                          {/* Rank Column */}
                          <td className="py-4.5 pl-4">
                            <span className={`font-graffiti text-xl ${badge.color}`}>
                              #{spitter.rank}
                            </span>
                          </td>

                          {/* Username Column */}
                          <td className="py-4.5">
                            <div className="flex items-center space-x-3">
                              <div className="h-10 w-10 rounded-full bg-raprank-maroon border border-raprank-cream/20 flex items-center justify-center font-graffiti text-sm text-white shrink-0">
                                {spitter.avatarText}
                              </div>
                              <div>
                                <span className="font-semibold tracking-wide text-white group-hover:text-raprank-neon transition-colors duration-150">
                                  {spitter.username}
                                </span>
                                {spitter.isCurrentUser && (
                                  <span className="ml-2 text-[10px] font-bold text-black bg-raprank-neon px-2 py-0.5 rounded-full uppercase tracking-wider">
                                    YOU
                                  </span>
                                )}
                              </div>
                            </div>
                          </td>

                          {/* Tracks Count Column */}
                          <td className="py-4.5">
                            <span className="font-bold text-white/80">{spitter.totalTracks}</span>
                          </td>

                          {/* Avg Score Column */}
                          <td className="py-4.5">
                            <span className="font-graffiti text-2xl text-white">{spitter.avgScore}</span>
                          </td>

                          {/* Grade Column */}
                          <td className="py-4.5">
                            <span className="font-graffiti text-xl text-raprank-neon">{spitter.grade}</span>
                          </td>

                          {/* Action Link Column */}
                          <td className="py-4.5 pr-4 text-right">
                            <Link
                              to="/profile"
                              className="inline-block text-xs font-bold tracking-widest uppercase text-raprank-skin/60 hover:text-raprank-neon hover:underline transition-colors duration-150"
                            >
                              VIEW PROFILE →
                            </Link>
                          </td>
                        </tr>
                      );
                    })}
                    
                    {(searchQuery !== "" && filteredSpitters.length === 0) && (
                      <tr>
                        <td colSpan={6} className="py-12 text-center text-sm font-semibold text-raprank-skin/40 uppercase tracking-widest">
                          No spitters found matching your search.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </main>

      <footer className="w-full py-6 text-center text-xs text-white/20 font-bold tracking-widest z-10 border-t border-raprank-maroon/10">
        © 2026 RAPRANK. BATTLES SCOREBOARD.
      </footer>
    </div>
  );
}
