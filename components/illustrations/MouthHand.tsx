import React from "react";

interface MouthHandProps extends React.SVGProps<SVGSVGElement> {
  className?: string;
  skinColor?: string;
  outlineColor?: string;
  interiorColor?: string;
  teethColor?: string;
  accentColor?: string;
}

export default function MouthHand({
  className = "",
  skinColor = "#e8c9a8",
  outlineColor = "#110c08",
  interiorColor = "#2a1518",
  teethColor = "#f5e9d8",
  accentColor = "#a8ff3e",
  ...props
}: MouthHandProps) {
  return (
    <svg
      viewBox="0 0 800 800"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`w-full h-full select-none ${className}`}
      aria-hidden="true"
      {...props}
    >
      {/* 1. Background Layer: Street Art Graffiti Splatters, Drips & Tag */}
      <g opacity="0.85">
        {/* Neon green paint splashes */}
        <path
          d="M 120 180 C 130 150, 160 140, 170 170 C 180 200, 150 220, 140 210 Z"
          fill={accentColor}
        />
        <circle cx="100" cy="150" r="8" fill={accentColor} />
        <circle cx="80" cy="200" r="5" fill={accentColor} />
        <circle cx="150" cy="120" r="12" fill={accentColor} />
        
        <path
          d="M 680 620 C 700 590, 730 600, 720 630 C 710 660, 670 650, 680 620 Z"
          fill={accentColor}
        />
        <circle cx="740" cy="650" r="6" fill={accentColor} />
        <circle cx="710" cy="680" r="10" fill={accentColor} />
        
        {/* Drips dropping down from face/mouth areas */}
        <path
          d="M 150 460 C 150 490, 155 510, 155 520 C 155 528, 148 528, 148 520 C 142 510, 142 490, 140 460 Z"
          fill={accentColor}
        />
        <path
          d="M 630 450 C 635 490, 640 510, 640 525 C 640 535, 632 535, 632 525 C 628 510, 625 490, 620 450 Z"
          fill={accentColor}
        />
        <circle cx="640" cy="545" r="4.5" fill={accentColor} />
      </g>

      {/* 2. Facial Context Lines (Cheek folds, Nose, Chin) */}
      <g stroke={outlineColor} strokeWidth="6" strokeLinecap="round">
        {/* Left Cheek / Smile Line */}
        <path d="M 140 260 Q 90 350 120 440" />
        
        {/* Right Cheek / Smile Line */}
        <path d="M 660 260 Q 710 350 680 440" />
        
        {/* Stylized Nose above the mouth */}
        <path d="M 370 145 Q 400 170 430 145" strokeWidth="8" />
        <path d="M 390 140 Q 400 130 410 140" strokeWidth="5" />
        
        {/* Chin crease below the lips */}
        <path d="M 360 630 Q 400 655 440 630" />
      </g>

      {/* 3. Main Yelling Mouth & Lips Structure */}
      <g>
        {/* Base Lips Shape (Warm Skin Tone #e8c9a8, thick outline) */}
        <path
          d="M 160 350 C 200 200, 300 170, 400 220 C 500 170, 600 200, 640 350 C 600 500, 520 590, 400 590 C 280 590, 200 500, 160 350 Z"
          fill={skinColor}
          stroke={outlineColor}
          strokeWidth="8"
          strokeLinejoin="round"
        />

        {/* Upper Lip Shadow */}
        <path
          d="M 172 342 C 210 230, 300 210, 400 240 C 500 210, 590 230, 628 342 C 610 310, 510 270, 400 270 C 290 270, 190 310, 172 342 Z"
          fill="#110c08"
          opacity="0.18"
        />

        {/* Gaping Mouth Cavity (Dark Interior #2a1518) */}
        <path
          d="M 200 350 C 240 270, 340 260, 400 260 C 460 260, 560 270, 600 350 C 560 460, 480 500, 400 500 C 320 500, 240 460, 200 350 Z"
          fill={interiorColor}
          stroke={outlineColor}
          strokeWidth="8"
          strokeLinejoin="round"
        />

        {/* Cream Teeth Sliver along the top arch */}
        <path
          d="M 220 330 C 260 290, 340 280, 400 280 C 460 280, 540 290, 580 330 L 565 355 C 520 335, 460 330, 400 330 C 340 330, 280 335, 235 355 Z"
          fill={teethColor}
          stroke={outlineColor}
          strokeWidth="5"
          strokeLinejoin="round"
        />

        {/* Individual Teeth Separation Lines */}
        <g stroke={outlineColor} strokeWidth="4" strokeLinecap="round">
          <line x1="290" y1="310" x2="295" y2="338" />
          <line x1="345" y1="298" x2="347" y2="330" />
          <line x1="400" y1="295" x2="400" y2="330" />
          <line x1="455" y1="298" x2="453" y2="330" />
          <line x1="510" y1="310" x2="505" y2="338" />
        </g>

        {/* Stylized Red/Pink Tongue at the bottom */}
        <path
          d="M 270 450 C 320 400, 360 395, 400 395 C 440 395, 480 400, 530 450 C 490 490, 450 495, 400 495 C 350 495, 310 490, 270 450 Z"
          fill="#d84b55"
          stroke={outlineColor}
          strokeWidth="5"
          strokeLinejoin="round"
        />
        
        {/* Tongue Center Crease Line */}
        <path
          d="M 400 420 L 400 480"
          stroke={outlineColor}
          strokeWidth="4"
          strokeLinecap="round"
        />
      </g>

      {/* 4. Hand holding Microphone Group (Rotated together to ensure 100% proper alignment) */}
      <g id="hand-and-microphone" transform="translate(300, 520) rotate(-16)">
        {/* Handle Shadow (drawn behind handle) */}
        <rect
          x="-20"
          y="0"
          width="48"
          height="240"
          rx="12"
          fill="#110c08"
          opacity="0.25"
          transform="translate(10, 10)"
        />
        
        {/* Microphone Handle */}
        <rect
          x="-24"
          y="0"
          width="48"
          height="240"
          rx="12"
          fill="#2a1518"
          stroke={outlineColor}
          strokeWidth="8"
        />
        
        {/* Neon Green Accent ring on mic */}
        <rect
          x="-29"
          y="15"
          width="58"
          height="16"
          rx="4"
          fill={accentColor}
          stroke={outlineColor}
          strokeWidth="6"
        />

        {/* Microphone Grille Capsule */}
        <circle
          cx="0"
          cy="-48"
          r="48"
          fill="#f0ece2"
          stroke={outlineColor}
          strokeWidth="8"
        />
        
        {/* Mesh Grid Lines on Grille */}
        <g stroke={outlineColor} strokeWidth="5" strokeLinecap="round">
          <path d="M -38 -48 Q 0 0 38 -48" />
          <path d="M -43 -30 Q 0 20 43 -30" />
          <path d="M -43 -66 Q 0 -20 43 -66" />
          
          <path d="M -20 -90 Q 15 -48 -20 -6" />
          <path d="M 0 -96 Q 35 -48 0 -2" />
          <path d="M 20 -90 Q 55 -48 20 -6" />
        </g>

        {/* Arm/Wrist (Skin Tone fill, bold outline) - Coming from bottom right of local coordinate space */}
        <path
          d="M 30 180 C 70 180, 120 220, 150 280 L 90 310 C 60 250, 20 220, -10 210 Z"
          fill={skinColor}
          stroke={outlineColor}
          strokeWidth="8"
          strokeLinejoin="round"
        />
        
        {/* Wrist Crease Lines */}
        <path
          d="M 55 205 Q 75 225 90 215"
          stroke={outlineColor}
          strokeWidth="5"
          strokeLinecap="round"
        />

        {/* Index Finger wrapping around handle */}
        <path
          d="M 24 45 C 55 45, 55 80, 24 80 L -20 80 C -38 80, -38 45, -20 45 Z"
          fill={skinColor}
          stroke={outlineColor}
          strokeWidth="8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Middle Finger wrapping around handle */}
        <path
          d="M 24 85 C 55 85, 55 120, 24 120 L -20 120 C -38 120, -38 85, -20 85 Z"
          fill={skinColor}
          stroke={outlineColor}
          strokeWidth="8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Ring Finger wrapping around handle */}
        <path
          d="M 24 125 C 55 125, 55 160, 24 160 L -20 160 C -38 160, -38 125, -20 125 Z"
          fill={skinColor}
          stroke={outlineColor}
          strokeWidth="8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Pinky Finger wrapping around handle */}
        <path
          d="M 24 165 C 55 165, 55 200, 24 200 L -20 200 C -38 200, -38 165, -20 165 Z"
          fill={skinColor}
          stroke={outlineColor}
          strokeWidth="8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Thumb wrapping over the front from the left */}
        <path
          d="M -24 95 C -45 95, -55 130, -30 145 C -10 155, 12 135, 12 115 C 12 95, -5 95, -24 95 Z"
          fill={skinColor}
          stroke={outlineColor}
          strokeWidth="8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        
        {/* Highlights on fingers (comic pop) */}
        <path
          d="M 32 53 C 42 53, 44 65, 34 70 Z"
          fill="#ffffff"
          opacity="0.35"
        />
        <path
          d="M 32 93 C 42 93, 44 105, 34 110 Z"
          fill="#ffffff"
          opacity="0.35"
        />
      </g>

      {/* Decorative dynamic soundwave lines around the yelling mouth */}
      <g stroke={outlineColor} strokeWidth="3.5" opacity="0.35" strokeDasharray="12 12" strokeLinecap="round">
        {/* Left soundwaves */}
        <path d="M 60 300 Q 30 350 60 400" />
        <path d="M 40 280 Q 5 350 40 420" strokeWidth="2.5" />
        
        {/* Right soundwaves */}
        <path d="M 740 300 Q 770 350 740 400" />
        <path d="M 760 280 Q 795 350 760 420" strokeWidth="2.5" />
      </g>
    </svg>
  );
}
