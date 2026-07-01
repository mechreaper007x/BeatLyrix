package handlers

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"raprank-upload/storage"
)


type UploadResponse struct {
	AudioURL string `json:"audio_url"`
	FileSize int64  `json:"file_size"`
	FileName string `json:"file_name"`
	Format   string `json:"format"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}

func generateUUID() (string, error) {
	b := make([]byte, 16)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:]), nil
}

func HandleUpload(w http.ResponseWriter, r *http.Request) {
	// Parse Multipart Form
	const maxFileSize = 15 * 1024 * 1024 // 15MB limit
	r.Body = http.MaxBytesReader(w, r.Body, maxFileSize)
	err := r.ParseMultipartForm(maxFileSize)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusRequestEntityTooLarge)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "File too large. Max size is 15MB."})
		return
	}

	// Retrieve file from "audio" field
	file, header, err := r.FormFile("audio")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "Invalid file in 'audio' field."})
		return
	}
	defer file.Close()

	// Validate Format (allowed: audio/mpeg or audio/wav)
	contentType := header.Header.Get("Content-Type")
	var format string
	if contentType == "audio/mpeg" || contentType == "audio/mp3" {
		format = "MP3"
	} else if contentType == "audio/wav" || contentType == "audio/x-wav" {
		format = "WAV"
	} else {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "Invalid file format. Only MP3 and WAV are allowed."})
		return
	}

	// Generate unique name
	uuid, err := generateUUID()
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "Internal server error."})
		return
	}
	ext := filepath.Ext(header.Filename)
	uniqueFilename := uuid + ext

	// Save to local storage
	savedPath, err := storage.SaveFile(file, uniqueFilename)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "Failed to save file: " + err.Error()})
		return
	}

	// Respond with metadata
	response := UploadResponse{
		AudioURL: savedPath,
		FileSize: header.Size,
		FileName: header.Filename,
		Format:   format,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}
