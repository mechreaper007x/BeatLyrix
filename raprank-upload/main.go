package main

import (
	"encoding/json"
	"log"
	"net/http"
	"raprank-upload/handlers"
)

func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "http://localhost:3000")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Credentials", "true")

		// Handle pre-flight request
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next(w, r)
	}
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[String]string{"status": "ok", "service": "raprank-upload"})
}

func main() {
	// Serve uploaded audio files statically
	// This enables playbacks of uploads at http://localhost:9090/uploads/audio/filename.mp3
	fs := http.FileServer(http.Dir("uploads/audio"))
	http.Handle("/uploads/audio/", http.StripPrefix("/uploads/audio/", fs))

	// API routes with CORS configuration
	http.HandleFunc("/upload", corsMiddleware(handlers.HandleUpload))
	http.HandleFunc("/health", corsMiddleware(healthHandler))

	log.Println("Go Audio Upload Service running on port 9090...")
	err := http.ListenAndServe(":9090", nil)
	if err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
