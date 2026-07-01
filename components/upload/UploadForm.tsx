"use client";

import React, { useState, useRef, useEffect } from "react";
import { Link } from "react-router-dom";

type Stage = "idle" | "uploading" | "analyzing" | "success" | "error";

export default function UploadForm() {
  const [title, setTitle] = useState("");
  const [lyrics, setLyrics] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [stage, setStage] = useState<Stage>("idle");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analysisText, setAnalysisText] = useState("Checking syllables...");
  const [newTrackId, setNewTrackId] = useState<number | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Clean up audio URL on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  // Syllables checking animation text loops during analysis stage
  useEffect(() => {
    if (stage !== "analyzing") return;
    const stages = [
      "Checking syllables & cadence...",
      "Analyzing multi-syllabic rhymes...",
      "Scoring alliteration & wordplay...",
      "Evaluating flow stability...",
    ];
    let idx = 0;
    const interval = setInterval(() => {
      idx = (idx + 1) % stages.length;
      setAnalysisText(stages[idx]);
    }, 1500);

    return () => clearInterval(interval);
  }, [stage]);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile: File) => {
    // Validate format (mp3, wav) and size (15MB)
    const validTypes = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav"];
    const fileExtension = selectedFile.name.split(".").pop()?.toLowerCase();
    const isValidType = validTypes.includes(selectedFile.type) || fileExtension === "mp3" || fileExtension === "wav";
    const isValidSize = selectedFile.size <= 15 * 1024 * 1024; // 15MB

    if (!isValidType) {
      alert("Invalid format! Please upload an MP3 or WAV audio file.");
      return;
    }

    if (!isValidSize) {
      alert("File is too large! Maximum allowed size is 15MB.");
      return;
    }

    // Set file
    setFile(selectedFile);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioUrl(URL.createObjectURL(selectedFile));
  };

  const removeFile = () => {
    setFile(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Words and character calculations
  const characterCount = lyrics.length;
  const wordCount = lyrics.trim() === "" ? 0 : lyrics.trim().split(/\s+/).length;

  const isFormValid = title.trim() !== "" && file !== null && lyrics.trim() !== "";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isFormValid || !file) return;

    setStage("uploading");
    setUploadProgress(10);

    try {
      // 1. Upload to Go Service (port 9090)
      const goFormData = new FormData();
      goFormData.append("audio", file);

      setUploadProgress(30);
      const goResponse = await fetch("http://localhost:9090/upload", {
        method: "POST",
        body: goFormData,
      });

      if (!goResponse.ok) {
        throw new Error("Failed to upload audio to file storage.");
      }

      setUploadProgress(70);
      const goData = await goResponse.json();
      const uploadedAudioUrl = goData.audio_url;

      setUploadProgress(100);
      
      // 2. Submit to Spring Boot (port 8080 via proxy)
      setStage("analyzing");
      setAnalysisText("Registering track metadata...");

      const token = localStorage.getItem("token");
      const response = await fetch("/api/tracks", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { "Authorization": `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          title: title,
          lyricsText: lyrics,
          audioUrl: uploadedAudioUrl,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to submit track to core registry.");
      }

      const trackData = await response.json();
      const trackId = trackData.id;
      setNewTrackId(trackId);

      // 3. Poll for analysis completion
      setAnalysisText("Triggering AI lyric scoring...");
      pollAnalysisStatus(trackId);

    } catch (err: any) {
      console.error("Upload/Analysis error:", err);
      setStage("error");
    }
  };

  const pollAnalysisStatus = (trackId: number) => {
    let attempts = 0;
    const maxAttempts = 30; // 45 seconds timeout
    
    const interval = setInterval(async () => {
      attempts++;
      if (attempts > maxAttempts) {
        clearInterval(interval);
        setStage("error");
        return;
      }

      try {
        const token = localStorage.getItem("token");
        const response = await fetch(`/api/tracks/${trackId}`, {
          headers: token ? { "Authorization": `Bearer ${token}` } : {},
        });

        if (response.ok) {
          const data = await response.json();
          if (data.totalScore !== null && data.scoreBreakdown !== null) {
            clearInterval(interval);
            setStage("success");
          }
        }
      } catch (err) {
        console.error("Error polling status:", err);
      }
    }, 1500);
  };

  const resetForm = () => {
    setTitle("");
    setLyrics("");
    removeFile();
    setNewTrackId(null);
    setStage("idle");
  };

  const retrySubmit = (e: React.FormEvent) => {
    handleSubmit(e);
  };

  // Render different screen components based on stages
  return (
    <div className="w-full max-w-2xl bg-black/75 backdrop-blur-2xl rounded-3xl p-6 md:p-8 border-3 border-raprank-neon shadow-[0_0_40px_rgba(168,255,62,0.15)] relative">
      
      {/* 1. IDLE STATE FORM */}
      {stage === "idle" && (
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Track Title */}
          <div>
            <label htmlFor="track-title" className="block font-graffiti text-xl tracking-wider text-raprank-neon mb-2 uppercase">
              Track Title
            </label>
            <input
              type="text"
              id="track-title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="ENTER TRACK TITLE"
              className="w-full bg-raprank-maroon/60 text-white placeholder-raprank-skin/40 font-semibold px-6 py-4 rounded-full border border-raprank-maroon/30 outline-none focus-visible:border-raprank-neon focus-visible:ring-4 focus-visible:ring-raprank-neon/30 transition-all duration-200"
            />
          </div>

          {/* Audio Upload Dropzone */}
          <div>
            <span className="block font-graffiti text-xl tracking-wider text-raprank-neon mb-2 uppercase">
              Audio File
            </span>
            
            {!file ? (
              <div
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`w-full min-h-[160px] border-3 border-dashed rounded-2xl flex flex-col items-center justify-center p-6 text-center cursor-pointer transition-all duration-300 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-raprank-neon/50 ${
                  isDragActive
                    ? "border-raprank-neon bg-raprank-neon/10 scale-[0.99]"
                    : "border-raprank-cream/30 bg-raprank-maroon/20 hover:border-raprank-neon/60 hover:bg-raprank-maroon/30"
                }`}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    fileInputRef.current?.click();
                  }
                }}
                aria-label="Upload audio track dropzone"
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept=".mp3,.wav,audio/mpeg,audio/wav"
                  className="hidden"
                  id="audio-file-input"
                />
                
                {/* Upload SVG Icon */}
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-10 w-10 text-raprank-neon mb-3 animate-pulse"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                  />
                </svg>

                <p className="text-sm font-semibold text-white/90">
                  Drag your audio file here or <span className="text-raprank-neon underline">click to browse</span>
                </p>
                <p className="text-xs font-semibold text-raprank-skin/50 mt-1">
                  MP3, WAV — Max 15MB
                </p>
              </div>
            ) : (
              /* File Loaded State */
              <div className="w-full bg-raprank-maroon/40 border border-raprank-maroon/50 rounded-2xl p-4 flex flex-col space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3 overflow-hidden">
                    {/* Music SVG Icon */}
                    <div className="bg-raprank-neon/10 p-2.5 rounded-lg shrink-0">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-6 w-6 text-raprank-neon"
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
                    <div className="overflow-hidden">
                      <p className="text-sm font-bold text-white truncate">{file.name}</p>
                      <p className="text-xs font-semibold text-raprank-skin/60">
                        {(file.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={removeFile}
                    className="p-1 rounded-full text-rose-400 hover:bg-rose-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400"
                    aria-label="Remove audio file"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-6 w-6"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                
                {/* Audio preview controls */}
                {audioUrl && (
                  <div className="pt-2 border-t border-raprank-maroon/20">
                    <span className="block text-xs font-bold text-raprank-neon uppercase mb-1.5">
                      Preview Player
                    </span>
                    <audio src={audioUrl} controls className="w-full h-8" />
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Lyrics Textarea */}
          <div>
            <div className="flex justify-between items-end mb-2">
              <label htmlFor="lyrics-textarea" className="block font-graffiti text-xl tracking-wider text-raprank-neon uppercase">
                Lyrics
              </label>
              <span className="text-xs font-semibold text-raprank-skin/50">
                PASTE YOUR BARS
              </span>
            </div>
            <textarea
              id="lyrics-textarea"
              rows={10}
              value={lyrics}
              onChange={(e) => setLyrics(e.target.value)}
              placeholder="Paste or type your lyrics here..."
              className="w-full bg-raprank-maroon/60 text-white placeholder-raprank-skin/40 font-semibold px-6 py-4 rounded-3xl border border-raprank-maroon/30 outline-none focus-visible:border-raprank-neon focus-visible:ring-4 focus-visible:ring-raprank-neon/30 transition-all duration-200 resize-y min-h-[160px]"
            />
            {/* Live counts */}
            <div className="flex justify-end space-x-4 text-xs font-bold text-raprank-skin/50 mt-1.5">
              <span>{characterCount} CHARACTERS</span>
              <span>•</span>
              <span>{wordCount} WORDS</span>
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={!isFormValid}
            className={`w-full py-4.5 px-6 font-graffiti text-2xl tracking-widest rounded-full transition-all duration-300 border-2 cursor-pointer outline-none focus-visible:ring-4 focus-visible:ring-raprank-neon focus-visible:ring-offset-2 focus-visible:ring-offset-black hover:scale-[1.01] active:scale-[0.99] ${
              isFormValid
                ? "bg-raprank-neon text-black border-raprank-neon shadow-lg shadow-raprank-neon/20 hover:shadow-raprank-neon/35"
                : "bg-white/10 text-white/30 border-transparent cursor-not-allowed"
            }`}
          >
            SUBMIT FOR SCORING
          </button>
        </form>
      )}

      {/* 2. UPLOADING STATE */}
      {stage === "uploading" && (
        <div className="py-12 flex flex-col items-center justify-center text-center space-y-6">
          {/* Animated Spinner Accent */}
          <div className="relative h-20 w-20 flex items-center justify-center">
            <div className="absolute inset-0 rounded-full border-4 border-raprank-neon/20 border-t-raprank-neon animate-spin shadow-[0_0_12px_#a8ff3e]" />
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-8 w-8 text-raprank-neon"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
              />
            </svg>
          </div>

          <div className="space-y-2">
            <h3 className="font-graffiti text-3xl text-white tracking-widest uppercase">
              UPLOADING TRACK...
            </h3>
            <p className="text-sm font-semibold text-raprank-skin/60">
              Sending your audio files to our ingestion services
            </p>
          </div>

          {/* Progress bar */}
          <div className="w-full max-w-md space-y-2">
            <div
              className="w-full bg-raprank-maroon/40 rounded-full h-4 overflow-hidden border border-raprank-maroon/20 relative"
              role="progressbar"
              aria-valuenow={uploadProgress}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                className="bg-raprank-neon h-full transition-all duration-300 shadow-[0_0_10px_#a8ff3e]"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <div className="flex justify-between text-xs font-bold text-raprank-neon tracking-widest">
              <span>PROGRESS</span>
              <span>{uploadProgress}%</span>
            </div>
          </div>
        </div>
      )}

      {/* 3. ANALYZING STATE (Custom bouncing equalizer bars animation) */}
      {stage === "analyzing" && (
        <div className="py-12 flex flex-col items-center justify-center text-center space-y-6">
          
          {/* Hip-hop bounce equalizer animation */}
          <div className="flex justify-center items-end space-x-2.5 h-20 my-4 select-none">
            <div className="w-3 bg-raprank-neon rounded-full animate-bounce h-12 shadow-[0_0_8px_#a8ff3e]" style={{ animationDuration: '0.8s', animationDelay: '0.1s' }} />
            <div className="w-3 bg-raprank-neon rounded-full animate-bounce h-20 shadow-[0_0_8px_#a8ff3e]" style={{ animationDuration: '0.7s', animationDelay: '0.3s' }} />
            <div className="w-3 bg-raprank-neon rounded-full animate-bounce h-14 shadow-[0_0_8px_#a8ff3e]" style={{ animationDuration: '0.9s', animationDelay: '0.0s' }} />
            <div className="w-3 bg-raprank-neon rounded-full animate-bounce h-18 shadow-[0_0_8px_#a8ff3e]" style={{ animationDuration: '0.6s', animationDelay: '0.4s' }} />
            <div className="w-3 bg-raprank-neon rounded-full animate-bounce h-10 shadow-[0_0_8px_#a8ff3e]" style={{ animationDuration: '0.8s', animationDelay: '0.2s' }} />
          </div>

          <div className="space-y-2">
            <h3 className="font-graffiti text-3xl text-white tracking-widest uppercase">
              ANALYZING YOUR BARS
            </h3>
            <p className="text-sm font-semibold text-raprank-neon uppercase tracking-widest animate-pulse h-6">
              {analysisText}
            </p>
            <p className="text-xs font-semibold text-raprank-skin/50 max-w-sm mx-auto">
              Our AI engine is measuring rhyme scheme density, syllable placement, syllable flow, and cadence structure.
            </p>
          </div>
        </div>
      )}

      {/* 4. SUCCESS RESULT STATE */}
      {stage === "success" && (
        <div className="py-6 flex flex-col items-center text-center space-y-8 animate-scaleIn">
          {/* Score Badge */}
          <div className="relative h-40 w-40 flex items-center justify-center">
            <div className="absolute inset-0 rounded-full border-4 border-raprank-neon animate-pulse shadow-[0_0_20px_#a8ff3e]" />
            <div className="flex flex-col items-center justify-center">
              <span className="font-graffiti text-6xl text-white drop-shadow-[0_4px_8px_rgba(168,255,62,0.4)]">
                87
              </span>
              <span className="text-xs font-bold text-raprank-neon tracking-widest uppercase">
                TOTAL SCORE
              </span>
            </div>
          </div>

          <div className="space-y-2">
            <h3 className="font-graffiti text-4xl text-white tracking-widest uppercase">
              TRACK ANALYZED SUCCESSFUL!
            </h3>
            <p className="text-sm font-semibold text-emerald-400 bg-emerald-950/40 border border-emerald-500/30 px-6 py-2.5 rounded-full inline-block">
              Successfully ingested & scored: <span className="font-bold text-white">"{title}"</span>
            </p>
          </div>

          {/* Quick Breakdown Summary */}
          <div className="w-full bg-raprank-maroon/30 border border-raprank-maroon/40 rounded-2xl p-6 grid grid-cols-3 gap-4">
            <div className="space-y-1">
              <span className="block text-xs font-bold text-raprank-skin/50 uppercase">RHYME DENSITY</span>
              <span className="font-graffiti text-2xl text-raprank-neon">A</span>
            </div>
            <div className="space-y-1 border-x border-raprank-maroon/20">
              <span className="block text-xs font-bold text-raprank-skin/50 uppercase">SYLLABLES</span>
              <span className="font-graffiti text-2xl text-raprank-neon">B+</span>
            </div>
            <div className="space-y-1">
              <span className="block text-xs font-bold text-raprank-skin/50 uppercase">FLOW STABILITY</span>
              <span className="font-graffiti text-2xl text-raprank-neon">A-</span>
            </div>
          </div>

          {/* Action buttons */}
          <div className="w-full space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Link
                to="/leaderboard"
                className="py-4 px-6 text-center font-graffiti text-xl tracking-widest rounded-full bg-raprank-neon text-black border-2 border-raprank-neon transition-all duration-300 hover:scale-[1.02] active:scale-[0.98] shadow-md shadow-raprank-neon/20"
              >
                VIEW ON LEADERBOARD
              </Link>
              <Link
                to={newTrackId ? `/tracks/${newTrackId}` : "/leaderboard"}
                className="py-4 px-6 text-center font-graffiti text-xl tracking-widest rounded-full bg-transparent text-raprank-neon border-2 border-raprank-neon transition-all duration-300 hover:scale-[1.02] active:scale-[0.98]"
              >
                VIEW FULL BREAKDOWN
              </Link>
            </div>
            
            <button
              type="button"
              onClick={resetForm}
              className="text-xs font-bold tracking-widest uppercase text-raprank-skin/50 hover:text-raprank-neon px-4 py-2 rounded-lg transition-colors duration-200"
            >
              UPLOAD ANOTHER TRACK
            </button>
          </div>
        </div>
      )}

      {/* 5. ERROR STATE */}
      {stage === "error" && (
        <div className="py-8 flex flex-col items-center justify-center text-center space-y-6" role="alert">
          {/* Warning SVG Icon */}
          <div className="h-16 w-16 bg-rose-950/40 border border-rose-500/30 rounded-full flex items-center justify-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-8 w-8 text-rose-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>

          <div className="space-y-2">
            <h3 className="font-graffiti text-3xl text-rose-400 tracking-widest uppercase">
              UPLOAD ERROR
            </h3>
            <p className="text-sm font-semibold text-rose-300/80 max-w-sm mx-auto">
              Ingestion service did not respond. Check your internet connection and verify that your Spring Boot auth endpoints are running.
            </p>
          </div>

          {/* Action buttons */}
          <div className="w-full max-w-md space-y-3 pt-2">
            <button
              type="button"
              onClick={retrySubmit}
              className="w-full py-4 px-6 font-graffiti text-xl tracking-widest bg-rose-600 hover:bg-rose-500 text-white rounded-full transition-all duration-300 hover:scale-[1.01] active:scale-[0.99] border-2 border-rose-500"
            >
              RETRY UPLOAD
            </button>
            
            <button
              type="button"
              onClick={() => setStage("idle")}
              className="w-full py-4 px-6 font-graffiti text-xl tracking-widest bg-transparent hover:bg-white/5 text-raprank-skin/70 rounded-full border-2 border-white/10 transition-all duration-300 hover:scale-[1.01] active:scale-[0.99]"
            >
              GO BACK TO EDIT FORM
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
