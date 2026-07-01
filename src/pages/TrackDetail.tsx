import React from "react";
import { useParams } from "react-router-dom";
import Navbar from "@/components/layout/Navbar";
import TrackDetail from "@/components/tracks/TrackDetail";

export default function TrackDetailPage() {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="min-h-screen w-full flex flex-col bg-raprank-dark text-white relative">
      <Navbar />
      
      {/* Dynamic wrapper passing track id */}
      <TrackDetail trackId={id || "1"} />
    </div>
  );
}
