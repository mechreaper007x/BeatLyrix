"use client";

import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";

interface Comment {
  id: string;
  username: string;
  text: string;
  timestamp: string;
  avatarText: string;
}

interface TrackDetailProps {
  trackId: string;
}

export default function TrackDetail({ trackId }: TrackDetailProps) {
  // State management
  const [track, setTrack] = useState<any>(null);
  const [comments, setComments] = useState<Comment[]>([]);
  const [liked, setLiked] = useState<boolean>(false);
  const [likeCount, setLikeCount] = useState<number>(0);
  const [newComment, setNewComment] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Load track and comments
  useEffect(() => {
    const fetchTrackData = async () => {
      setLoading(true);
      setError(null);
      try {
        const token = localStorage.getItem("token");
        const headers: HeadersInit = token ? { "Authorization": `Bearer ${token}` } : {};

        // Fetch track
        const trackRes = await fetch(`/api/tracks/${trackId}`, { headers });
        if (!trackRes.ok) {
          throw new Error("Failed to load track details");
        }
        const trackData = await trackRes.json();
        setTrack(trackData);
        setLikeCount(trackData.likeCount);

        // check if user has liked locally
        const activeUser = localStorage.getItem("username");
        if (activeUser) {
          const userLiked = localStorage.getItem(`liked_${activeUser}_${trackId}`) === "true";
          setLiked(userLiked);
        }

        // Fetch comments
        const commentsRes = await fetch(`/api/tracks/${trackId}/comments`, { headers });
        if (commentsRes.ok) {
          const commentsData = await commentsRes.json();
          // Map backend CommentResponse to frontend Comment struct
          const mappedComments = commentsData.map((c: any) => ({
            id: c.id.toString(),
            username: c.artistUsername,
            text: c.content,
            timestamp: new Date(c.createdAt).toLocaleDateString() || "Just now",
            avatarText: c.artistUsername.substring(0, 2).toUpperCase(),
          }));
          setComments(mappedComments);
        }
      } catch (err: any) {
        console.error(err);
        setError(err.message || "Failed to load data.");
      } finally {
        setLoading(false);
      }
    };

    if (trackId) {
      fetchTrackData();
    }
  }, [trackId]);

  // Handlers
  const handleLikeToggle = async () => {
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        alert("Please login to like tracks!");
        return;
      }

      const response = await fetch(`/api/tracks/${trackId}/like`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        setLiked(data.liked);
        setLikeCount(data.likeCount);

        // Sync local storage state
        const activeUser = localStorage.getItem("username");
        if (activeUser) {
          localStorage.setItem(`liked_${activeUser}_${trackId}`, data.liked ? "true" : "false");
        }
      }
    } catch (err) {
      console.error("Failed to toggle like:", err);
    }
  };

  const handlePostComment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newComment.trim()) return;

    try {
      const token = localStorage.getItem("token");
      if (!token) {
        alert("Please login to comment!");
        return;
      }

      const response = await fetch(`/api/tracks/${trackId}/comments`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ content: newComment.trim() }),
      });

      if (response.ok) {
        const c = await response.json();
        const newCommentObj: Comment = {
          id: c.id.toString(),
          username: c.artistUsername,
          text: c.content,
          timestamp: "Just now",
          avatarText: c.artistUsername.substring(0, 2).toUpperCase(),
        };

        setComments((prev) => [newCommentObj, ...prev]);
        setNewComment("");
      }
    } catch (err) {
      console.error("Failed to post comment:", err);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen w-full flex items-center justify-center bg-raprank-dark text-white font-graffiti text-3xl">
        LOADING TRACK METRICS...
      </div>
    );
  }

  if (error || !track) {
    return (
      <div className="min-h-screen w-full flex flex-col items-center justify-center bg-raprank-dark text-white p-6">
        <h2 className="font-graffiti text-4xl text-rose-500 mb-4">ERROR</h2>
        <p className="font-sans text-lg text-raprank-skin/80 mb-6">{error || "Track not found."}</p>
        <Link to="/leaderboard" className="font-graffiti text-xl text-raprank-neon hover:underline">
          GO TO LEADERBOARD
        </Link>
      </div>
    );
  }

  // Lyrical metrics helper
  const lyricsLines = track.lyricsText.split("\n");
  const wordCount = track.lyricsText.trim().split(/\s+/).filter((w: string) => w.trim() !== "").length;

  const syllableScore = track.scoreBreakdown?.syllableScore ?? 0;
  const alliterationScore = track.scoreBreakdown?.alliterationScore ?? 0;
  const flowScore = track.scoreBreakdown?.flowScore ?? 0;
  const totalScore = track.totalScore || 0;
  const grade = track.grade || "PENDING";

  // New metrics
  const rhymeScore = track.scoreBreakdown?.rhymeScore ?? track.scoreBreakdown?.rhyme_score ?? 0;
  const wordplayScore = track.scoreBreakdown?.wordplayScore ?? track.scoreBreakdown?.wordplay_score ?? 0;
  const syllableWeight = track.scoreBreakdown?.syllableWeight ?? track.scoreBreakdown?.syllable_weight ?? 0;
  const vocabularyUniqueness = track.scoreBreakdown?.vocabularyUniqueness ?? track.scoreBreakdown?.vocabulary_uniqueness ?? 0;

  const doubleEntendresCount = track.scoreBreakdown?.doubleEntendresCount ?? track.scoreBreakdown?.double_entendres_count ?? 0;
  const punsCount = track.scoreBreakdown?.punsCount ?? track.scoreBreakdown?.puns_count ?? 0;
  const similesCount = track.scoreBreakdown?.similesCount ?? track.scoreBreakdown?.similes_count ?? 0;
  const metaphorsCount = track.scoreBreakdown?.metaphorsCount ?? track.scoreBreakdown?.metaphors_count ?? 0;
  const wordplayExplanation = track.scoreBreakdown?.wordplayExplanation ?? track.scoreBreakdown?.wordplay_explanation;

  return (
    <div className="w-full flex flex-col bg-[#1c1410] text-[#ffffff] min-h-screen">
      {/* Load Tabler Icons webfont CDN */}
      <link
        rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css"
      />

      {/* Grid Pattern Backdrop */}
      <div className="absolute inset-0 bg-[radial-gradient(#2a1518_1.5px,transparent_1.5px)] [background-size:24px_24px] opacity-25 pointer-events-none z-0" />

      <div className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 flex flex-col gap-8">
        
        {/* SECTION 1 — Top area: Track identity (full width) */}
        <section className="w-full bg-[#111111] border border-[#2a2a2a] rounded-3xl p-6 md:p-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-6 shadow-2xl">
          {/* Left side */}
          <div className="w-full md:w-3/5 flex flex-col items-start">
            <h1 className="font-graffiti text-5xl text-white tracking-wide uppercase leading-tight select-text">
              {track.title}
            </h1>
            <p className="text-lg font-sans text-[#888888] mt-1.5">
              by{" "}
              <Link
                to="/profile"
                className="text-[#888888] hover:text-[#a8ff3e] transition-colors duration-150 underline decoration-dotted font-semibold"
              >
                {track.artistUsername}
              </Link>
            </p>
            <p className="text-xs font-sans text-[#888888] mt-1 font-medium">
              Released: {new Date(track.createdAt).toLocaleDateString()}
            </p>

            <button
              type="button"
              onClick={handleLikeToggle}
              aria-label="Like this track"
              aria-pressed={liked}
              className={`group flex items-center gap-2 mt-4 px-6 py-2.5 rounded-full border text-sm font-bold tracking-wider uppercase transition-all duration-200 cursor-pointer active:scale-95 ${
                liked
                  ? "bg-red-950/40 border-red-500/80 text-red-400 shadow-[0_0_12px_rgba(239,68,68,0.2)]"
                  : "bg-[#1a1a1a] hover:bg-[#2a2a2a] border-[#2a2a2a] text-white hover:border-[#a8ff3e]/40"
              }`}
            >
              <i
                className={`ti ${
                  liked ? "ti-heart-filled text-red-500 scale-110" : "ti-heart text-white/70 group-hover:text-[#a8ff3e]"
                } transition-transform duration-200`}
              />
              <span>{likeCount} Likes</span>
            </button>
          </div>

          {/* Right side */}
          <div className="w-full md:w-2/5 flex items-center justify-start md:justify-end gap-6 border-t border-[#2a2a2a]/60 pt-6 md:pt-0 md:border-t-0">
            <div className="flex flex-col items-end">
              <span className="font-graffiti text-xs text-[#888888] tracking-widest uppercase">
                SCORE
              </span>
              <div className="flex items-baseline leading-none mt-1">
                <span className="font-graffiti text-8xl text-[#a8ff3e] select-text">
                  {totalScore}
                </span>
                <span className="font-graffiti text-2xl text-[#888888] ml-1">
                  /100
                </span>
              </div>
            </div>

            {/* Large Grade Badge */}
            <div className="flex items-center justify-center h-20 w-20 rounded-2xl bg-[#1a1a1a] border-2 border-[#a8ff3e] text-[#a8ff3e] shadow-[0_0_15px_rgba(168,255,62,0.15)] select-none">
              <span className="font-graffiti text-5xl leading-none">
                {grade}
              </span>
            </div>
          </div>
        </section>

        {/* SECTION 2 — Audio player */}
        {track.audioUrl ? (
          <section className="w-full bg-[#111111] border border-[#2a2a2a] border-l-4 border-l-[#a8ff3e] rounded-2xl p-5 shadow-lg">
            <audio
              controls
              src={track.audioUrl}
              className="w-full focus:outline-none"
              style={{ accentColor: "#a8ff3e" }}
            >
              Your browser does not support the audio element.
            </audio>
            
            <div className="flex gap-2 mt-3.5">
              <span className="text-[10px] font-bold text-[#888888] bg-[#1a1a1a] border border-[#2a2a2a] px-3 py-1 rounded-full uppercase tracking-wider">
                AUDIO STREAM
              </span>
            </div>
          </section>
        ) : (
          <section className="w-full bg-[#111111]/40 border border-[#2a2a2a] border-l-4 border-l-gray-600 rounded-2xl p-5 shadow-lg flex items-center justify-between">
            <span className="text-sm font-sans text-[#888888]">No audio track uploaded. Lyrical scoring was calculated in Text-Only mode.</span>
            <span className="text-[10px] font-bold text-gray-400 bg-[#1a1a1a] border border-[#2a2a2a] px-3 py-1 rounded-full uppercase tracking-wider">
              TEXT ONLY
            </span>
          </section>
        )}

        {/* SECTION 3 — Two column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left column - Lyrics */}
          <section className="lg:col-span-7 w-full flex flex-col">
            <h2 className="font-graffiti text-2xl text-[#a8ff3e] tracking-wider mb-4 uppercase">
              LYRICS
            </h2>
            <div className="bg-[#2a1518] border border-raprank-maroon/20 rounded-2xl p-5 md:p-6 shadow-inner">
              <div className="max-h-96 overflow-y-auto pr-2 custom-scrollbar">
                <p className="font-sans text-sm text-white leading-relaxed whitespace-pre-wrap select-text">
                  {track.lyricsText}
                </p>
              </div>
            </div>
            <p className="text-xs text-[#888888] mt-3 font-semibold tracking-wider font-sans">
              {lyricsLines.length} lines • {wordCount} words
            </p>
          </section>

          {/* Right column - Score Breakdown */}
          <section className="lg:col-span-5 w-full flex flex-col">
            <h2 className="font-graffiti text-2xl text-[#a8ff3e] tracking-wider mb-4 uppercase">
              SCORE BREAKDOWN
            </h2>
            
            <div className="bg-[#111111] border border-[#2a2a2a] rounded-2xl p-6 flex flex-col gap-6 shadow-lg">
              
              {/* Syllable Density */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs font-bold text-white uppercase tracking-wider">
                  <span>Syllable Density</span>
                  <span className="text-[#a8ff3e] font-sans">{syllableScore}%</span>
                </div>
                <div 
                  role="progressbar"
                  aria-valuenow={syllableScore}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  className="w-full bg-[#2a1518] rounded-full h-2.5 overflow-hidden border border-[#2a1518]/60"
                >
                  <div
                    className="bg-[#a8ff3e] h-full rounded-full shadow-[0_0_8px_#a8ff3e] transition-all duration-300"
                    style={{ width: `${syllableScore}%` }}
                  />
                </div>
              </div>

              {/* Alliteration Score */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs font-bold text-white uppercase tracking-wider">
                  <span>Alliteration Score</span>
                  <span className="text-[#a8ff3e] font-sans">{alliterationScore}%</span>
                </div>
                <div 
                  role="progressbar"
                  aria-valuenow={alliterationScore}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  className="w-full bg-[#2a1518] rounded-full h-2.5 overflow-hidden border border-[#2a1518]/60"
                >
                  <div
                    className="bg-[#a8ff3e] h-full rounded-full shadow-[0_0_8px_#a8ff3e] transition-all duration-300"
                    style={{ width: `${alliterationScore}%` }}
                  />
                </div>
              </div>

              {/* Flow Complexity */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs font-bold text-white uppercase tracking-wider">
                  <span>Flow Complexity</span>
                  <span className="text-[#a8ff3e] font-sans">
                    {track.audioUrl ? `${flowScore}%` : "N/A (Text-Only)"}
                  </span>
                </div>
                {track.audioUrl && (
                  <div 
                    role="progressbar"
                    aria-valuenow={flowScore}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    className="w-full bg-[#2a1518] rounded-full h-2.5 overflow-hidden border border-[#2a1518]/60"
                  >
                    <div
                      className="bg-[#a8ff3e] h-full rounded-full shadow-[0_0_8px_#a8ff3e] transition-all duration-300"
                      style={{ width: `${flowScore}%` }}
                    />
                  </div>
                )}
              </div>

              {/* Rhyme Complexity */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs font-bold text-white uppercase tracking-wider">
                  <span>Rhyme Complexity</span>
                  <span className="text-[#a8ff3e] font-sans">{rhymeScore}%</span>
                </div>
                <div 
                  role="progressbar"
                  aria-valuenow={rhymeScore}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  className="w-full bg-[#2a1518] rounded-full h-2.5 overflow-hidden border border-[#2a1518]/60"
                >
                  <div
                    className="bg-[#a8ff3e] h-full rounded-full shadow-[0_0_8px_#a8ff3e] transition-all duration-300"
                    style={{ width: `${rhymeScore}%` }}
                  />
                </div>
              </div>

              {/* Wordplay Score */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs font-bold text-white uppercase tracking-wider">
                  <span>Wordplay Score</span>
                  <span className="text-[#a8ff3e] font-sans">{wordplayScore}%</span>
                </div>
                <div 
                  role="progressbar"
                  aria-valuenow={wordplayScore}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  className="w-full bg-[#2a1518] rounded-full h-2.5 overflow-hidden border border-[#2a1518]/60"
                >
                  <div
                    className="bg-[#a8ff3e] h-full rounded-full shadow-[0_0_8px_#a8ff3e] transition-all duration-300"
                    style={{ width: `${wordplayScore}%` }}
                  />
                </div>
              </div>

              {/* Syllable Weight (Vocabulary) */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs font-bold text-white uppercase tracking-wider">
                  <span>Syllable Weight (Vocabulary)</span>
                  <span className="text-[#a8ff3e] font-sans">{syllableWeight}%</span>
                </div>
                <div 
                  role="progressbar"
                  aria-valuenow={syllableWeight}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  className="w-full bg-[#2a1518] rounded-full h-2.5 overflow-hidden border border-[#2a1518]/60"
                >
                  <div
                    className="bg-[#a8ff3e] h-full rounded-full shadow-[0_0_8px_#a8ff3e] transition-all duration-300"
                    style={{ width: `${syllableWeight}%` }}
                  />
                </div>
              </div>

              {/* Lyrical Statistics */}
              <div className="border-t border-[#2a2a2a]/60 pt-4 mt-2">
                <h3 className="text-xs font-bold text-white uppercase tracking-wider mb-3">Lyrical Breakdown</h3>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="flex justify-between bg-[#1a1a1a] p-2 rounded-lg border border-[#2a2a2a]">
                    <span className="text-[#888888]">Similes</span>
                    <span className="text-white font-bold">{similesCount}</span>
                  </div>
                  <div className="flex justify-between bg-[#1a1a1a] p-2 rounded-lg border border-[#2a2a2a]">
                    <span className="text-[#888888]">Metaphors</span>
                    <span className="text-white font-bold">{metaphorsCount}</span>
                  </div>
                  <div className="flex justify-between bg-[#1a1a1a] p-2 rounded-lg border border-[#2a2a2a]">
                    <span className="text-[#888888]">Puns / Homophones</span>
                    <span className="text-white font-bold">{punsCount}</span>
                  </div>
                  <div className="flex justify-between bg-[#1a1a1a] p-2 rounded-lg border border-[#2a2a2a]">
                    <span className="text-[#888888]">Double Entendres</span>
                    <span className="text-white font-bold">{doubleEntendresCount}</span>
                  </div>
                  <div className="flex justify-between bg-[#1a1a1a] p-2 rounded-lg border border-[#2a2a2a] col-span-2">
                    <span className="text-[#888888]">Lexical Diversity (TTR)</span>
                    <span className="text-white font-bold">{(vocabularyUniqueness * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* Wordplay Explanation Accordion */}
              {wordplayExplanation && (
                <div className="bg-[#1a1a1a] border border-[#2a2a2a] hover:border-[#a8ff3e]/40 rounded-2xl p-4.5 mt-4 space-y-3 transition-colors duration-300 shadow-lg">
                  <div className="flex items-center gap-2 border-b border-[#2a2a2a]/80 pb-2.5">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-4 w-4 text-[#a8ff3e]"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                      />
                    </svg>
                    <h4 className="text-xs font-bold text-[#a8ff3e] uppercase tracking-wider">
                      Wordplay Critique & Analysis
                    </h4>
                  </div>
                  <div className="text-xs text-rose-100/80 font-sans leading-relaxed whitespace-pre-line select-text max-h-72 overflow-y-auto pr-1">
                    {wordplayExplanation}
                  </div>
                </div>
              )}

              <p className="text-xs text-[#888888] italic font-sans mt-1 leading-normal">
                Scores are AI-generated based on lyric and beat analysis.
              </p>

              {/* Summary Row */}
              <div className="bg-[#1a1a1a] border border-[#2a2a2a] hover:border-[#a8ff3e] rounded-2xl p-4.5 flex justify-between items-center mt-3 shadow-md transition-colors duration-300">
                <span className="font-graffiti text-lg text-white tracking-widest uppercase">
                  TOTAL SCORE
                </span>
                <div className="flex items-baseline leading-none">
                  <span className="font-graffiti text-3xl text-[#a8ff3e] select-text">
                    {totalScore}
                  </span>
                  <span className="font-graffiti text-lg text-[#888888] ml-1">
                    /100
                  </span>
                </div>
              </div>

            </div>
          </section>
        </div>

        {/* SECTION 4 — Comments section (full width) */}
        <section className="w-full mt-6">
          <h2 className="font-graffiti text-3xl text-white tracking-wider mb-6 uppercase">
            THE CYPHER
          </h2>

          {/* Comment input area */}
          <form onSubmit={handlePostComment} className="w-full bg-[#111111] border border-[#2a2a2a] rounded-3xl p-5 md:p-6 shadow-lg mb-8 flex flex-col gap-3">
            <div className="w-full">
              <label htmlFor="comment-textarea" className="sr-only">
                Drop your thoughts on this track
              </label>
              <textarea
                id="comment-textarea"
                rows={3}
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                placeholder="Drop your thoughts..."
                className="w-full bg-[#2a1518] border border-raprank-maroon/20 text-white placeholder-white/30 p-4 rounded-2xl outline-none focus:ring-2 focus:ring-[#a8ff3e]/40 focus:border-[#a8ff3e] transition-all duration-200 resize-none font-sans text-sm"
              />
            </div>
            
            <button
              type="submit"
              className="self-end px-8 py-2.5 rounded-full bg-[#a8ff3e] hover:bg-[#96eb34] active:scale-97 text-black font-bold tracking-widest uppercase text-xs transition-all duration-200 cursor-pointer shadow-[0_4px_12px_rgba(168,255,62,0.15)]"
            >
              POST
            </button>
          </form>

          {/* Comments List */}
          <div className="bg-[#111111] border border-[#2a2a2a] rounded-3xl p-6 shadow-xl flex flex-col">
            {comments.length === 0 ? (
              <div className="text-center py-10 text-[#888888] font-sans font-semibold">
                No bars dropped yet. Be the first.
              </div>
            ) : (
              <div className="flex flex-col">
                {comments.map((comment) => (
                  <div
                    key={comment.id}
                    className="flex items-start gap-4 py-5 border-b border-[#2a2a2a] last:border-b-0"
                  >
                    {/* Glowing Initials Avatar */}
                    <div className="h-10 w-10 rounded-full bg-[#2a1518] border border-[#a8ff3e] text-[#a8ff3e] font-graffiti text-sm flex items-center justify-center shrink-0 shadow-[0_0_8px_rgba(168,255,62,0.15)] select-none">
                      {comment.avatarText}
                    </div>

                    <div className="flex-grow">
                      <div className="flex justify-between items-baseline mb-1 gap-2">
                        <span className="font-sans font-bold text-white text-sm">
                          {comment.username}
                        </span>
                        <span className="font-sans text-[10px] text-[#888888] font-semibold">
                          {comment.timestamp}
                        </span>
                      </div>
                      <p className="font-sans text-sm text-white/90 leading-relaxed break-words whitespace-pre-wrap select-text">
                        {comment.text}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

      </div>
    </div>
  );
}
