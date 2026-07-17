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
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [estimatedTime, setEstimatedTime] = useState<string>("Calculating...");
  const [newTrackId, setNewTrackId] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  const [resultScore, setResultScore] = useState<number | null>(null);
  const [resultRhyme, setResultRhyme] = useState<number | null>(null);
  const [resultSyllable, setResultSyllable] = useState<number | null>(null);
  const [resultFlow, setResultFlow] = useState<number | null>(null);

  const getGrade = (score: number | null) => {
    if (score === null) return "N/A";
    if (score >= 90) return "S";
    if (score >= 80) return "A";
    if (score >= 70) return "B";
    if (score >= 60) return "C";
    return "D";
  };
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Clean up audio URL on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  // Analysis text is set dynamically via real-time polling updates from the backend

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

  const isFormValid = title.trim() !== "" && lyrics.trim() !== "";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isFormValid) return;

    setStage("uploading");
    setUploadProgress(10);
    setErrorMessage(null);

    try {
      let uploadedAudioUrl = null;

      if (file) {
        // 1. Upload to Go Service (port 9090)
        const goFormData = new FormData();
        goFormData.append("audio", file);
        goFormData.append("lyrics", lyrics);

        setUploadProgress(30);
        const goResponse = await fetch("http://localhost:9090/upload-pipeline", {
          method: "POST",
          body: goFormData,
        });

        if (!goResponse.ok) {
          throw new Error("Failed to upload audio to file storage service.");
        }

        setUploadProgress(70);
        const goData = await goResponse.json();
        uploadedAudioUrl = goData.audio_url;
      }

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
        throw new Error("Failed to submit track to core registry service.");
      }

      const trackData = await response.json();
      const trackId = trackData.id;
      setNewTrackId(trackId);

      // 3. Poll for analysis completion
      setAnalysisText("Triggering AI lyric scoring...");
      pollAnalysisStatus(trackId);

    } catch (err: any) {
      console.error("Upload/Analysis error:", err);
      setErrorMessage(err.message || String(err));
      setStage("error");
    }
  };

  const pollAnalysisStatus = (trackId: number) => {
    let attempts = 0;
    const maxAttempts = 2400; // 1 hour timeout (at 1.5s intervals)
    setAnalysisProgress(5);
    setEstimatedTime("Calculating...");
    
    // Variables to track simulated progress increments per status
    let analyzingTextProgress = 30;

    const interval = setInterval(async () => {
      attempts++;
      if (attempts > maxAttempts) {
        clearInterval(interval);
        setErrorMessage("Analysis polling timed out (1h). The external AI service took too long.");
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
          if (data.status === "FAILED") {
            clearInterval(interval);
            setErrorMessage("Lyric analysis failed in background. Check backend logs.");
            setStage("error");
          } else if (data.totalScore !== null && data.scoreBreakdown !== null) {
            setResultScore(Math.round(data.totalScore));
            setResultRhyme(data.scoreBreakdown.rhymeScore !== undefined && data.scoreBreakdown.rhymeScore !== null ? Math.round(data.scoreBreakdown.rhymeScore) : null);
            setResultSyllable(data.scoreBreakdown.syllableScore !== undefined && data.scoreBreakdown.syllableScore !== null ? Math.round(data.scoreBreakdown.syllableScore) : null);
            setResultFlow(data.scoreBreakdown.flowScore !== undefined && data.scoreBreakdown.flowScore !== null ? Math.round(data.scoreBreakdown.flowScore) : null);

            setAnalysisProgress(100);
            setEstimatedTime("Complete!");
            clearInterval(interval);
            setTimeout(() => {
              setStage("success");
            }, 500);
          } else {
            // Dynamically set progress and time estimation based on backend status
            if (data.status === "PENDING") {
              setAnalysisText("Registering track metadata...");
              setAnalysisProgress(15);
              setEstimatedTime("~5 seconds remaining");
            } else if (data.status === "ANALYZING_TEXT") {
              setAnalysisText("Compiling lyrics: lexing syllables and calculating transition entropy...");
              if (analyzingTextProgress < 98) {
                analyzingTextProgress += 4.0;
              }
              setAnalysisProgress(Math.min(Math.round(analyzingTextProgress), 98));
              setEstimatedTime("~2s remaining");
            } else {
              setAnalysisText("Compiling lyrical quality metrics...");
              setAnalysisProgress(50);
              setEstimatedTime("Processing...");
            }
          }
        } else {
          throw new Error(`Failed response from core backend: ${response.statusText}`);
        }
      } catch (err: any) {
        console.error("Error polling status:", err);
      }
    }, 1500);
  };

  const resetForm = () => {
    setTitle("");
    setLyrics("");
    removeFile();
    setNewTrackId(null);
    setErrorMessage(null);
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

          {/* Audio Upload Dropzone Removed for LQI Lyrical Decoupling */}

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

          <div className="space-y-2 w-full max-w-md">
            <h3 className="font-graffiti text-3xl text-white tracking-widest uppercase">
              ANALYZING YOUR BARS
            </h3>
            <p className="text-sm font-semibold text-raprank-neon uppercase tracking-widest animate-pulse h-6">
              {analysisText}
            </p>
            
            {/* Dynamic Progress Bar */}
            <div className="pt-2">
              <div
                className="w-full bg-raprank-maroon/40 rounded-full h-4 overflow-hidden border border-raprank-maroon/20 relative"
                role="progressbar"
                aria-valuenow={analysisProgress}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className="bg-raprank-neon h-full transition-all duration-300 shadow-[0_0_10px_#a8ff3e]"
                  style={{ width: `${analysisProgress}%` }}
                />
              </div>
              <div className="flex justify-between text-xs font-bold text-raprank-neon tracking-widest mt-2">
                <span>{estimatedTime}</span>
                <span>{analysisProgress}%</span>
              </div>
            </div>

            <p className="text-xs font-semibold text-raprank-skin/50 max-w-sm mx-auto pt-4">
              Our lexical compiler is tokenizing bars, scanning syllables with schwa-deletion, and calculating rhyme state transition entropy.
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
                {resultScore ?? "--"}
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
              <span className="block text-xs font-bold text-raprank-skin/50 uppercase">RHYME COMPLEXITY</span>
              <span className="font-graffiti text-2xl text-raprank-neon">
                {resultRhyme !== null ? `${resultRhyme} (${getGrade(resultRhyme)})` : "N/A"}
              </span>
            </div>
            <div className="space-y-1 border-x border-raprank-maroon/20">
              <span className="block text-xs font-bold text-raprank-skin/50 uppercase">SYLLABLE DENSITY</span>
              <span className="font-graffiti text-2xl text-raprank-neon">
                {resultSyllable !== null ? `${resultSyllable} (${getGrade(resultSyllable)})` : "N/A"}
              </span>
            </div>
            <div className="space-y-1">
              <span className="block text-xs font-bold text-raprank-skin/50 uppercase">FLOW STABILITY</span>
              <span className="font-graffiti text-2xl text-raprank-neon">
                {resultFlow !== null ? `${resultFlow} (${getGrade(resultFlow)})` : "N/A"}
              </span>
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
            {errorMessage && (
              <div className="mt-4 p-3 bg-red-950/30 border border-red-500/20 rounded-xl text-xs font-mono text-rose-400 max-w-md mx-auto select-text break-all">
                <span className="font-bold uppercase block mb-1">Error Debugger Log:</span>
                {errorMessage}
              </div>
            )}
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
