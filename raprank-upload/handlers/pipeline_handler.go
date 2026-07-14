package handlers

import (
	"encoding/json"
	"net/http"
	"path/filepath"
	"raprank-upload/storage"
)

// PipelineUploadResponse mirrors UploadResponse (see upload.go). It used to
// also carry AlignedLyrics/MistralPayload from two extra downstream calls
// (a local Python "heuristic transcriber" and a hardcoded-localhost Whisper
// call) that raced ahead of the real scoring pipeline for no reason: nothing
// consumed those fields (the frontend only reads audio_url), the Whisper
// call target was wrong for any non-local deployment, and neither call had
// a timeout -- so every upload paid for two broken, hang-prone requests
// before it could even respond. Removed; actual transcription/scoring goes
// through Spring Boot -> raprank-nlp, which calls the real Whisper Space via
// transcription_service.py.
type PipelineUploadResponse struct {
	AudioURL string `json:"audio_url"`
	FileSize int64  `json:"file_size"`
	FileName string `json:"file_name"`
	Format   string `json:"format"`
}

func HandlePipelineUpload(w http.ResponseWriter, r *http.Request) {
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

	// Retrieve audio file
	file, header, err := r.FormFile("audio")
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "Invalid file in 'audio' field."})
		return
	}
	defer file.Close()

	// Retrieve lyrics
	lyrics := r.FormValue("lyrics")
	if lyrics == "" {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "Missing 'lyrics' field."})
		return
	}

	// Generate unique name and save file
	uuid, err := generateUUID()
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "Internal server error."})
		return
	}
	ext := filepath.Ext(header.Filename)
	uniqueFilename := uuid + ext

	savedPath, err := storage.SaveFile(file, uniqueFilename)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "Failed to save file: " + err.Error()})
		return
	}

	response := PipelineUploadResponse{
		AudioURL: savedPath,
		FileSize: header.Size,
		FileName: header.Filename,
		Format:   filepath.Ext(header.Filename),
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}
