package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"raprank-upload/storage"
)

// PipelineUploadResponse is the extended response including formatting data
type PipelineUploadResponse struct {
	AudioURL       string      `json:"audio_url"`
	FileSize       int64       `json:"file_size"`
	FileName       string      `json:"file_name"`
	Format         string      `json:"format"`
	AlignedLyrics  interface{} `json:"aligned_lyrics,omitempty"`
	MistralPayload interface{} `json:"mistral_payload,omitempty"`
}

func HandlePipelineUpload(w http.ResponseWriter, r *http.Request) {
	// 1. Parse Multipart Form
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
	
	// Reset file pointer so we can read it again for HTTP requests
	file.Seek(0, 0)

	// =========================================================================
	// 2. Call Python Heuristic Transcriber (localhost:8001/sync)
	// =========================================================================
	var pyResponseBody interface{}
	
	bodyBuf := &bytes.Buffer{}
	bodyWriter := multipart.NewWriter(bodyBuf)
	
	// Add lyrics
	bodyWriter.WriteField("lyrics", lyrics)
	
	// Add audio
	fileWriter, err := bodyWriter.CreateFormFile("file", header.Filename)
	if err == nil {
		io.Copy(fileWriter, file)
	}
	bodyWriter.Close()
	
	pyReq, err := http.NewRequest("POST", "http://localhost:8001/sync", bodyBuf)
	if err == nil {
		pyReq.Header.Set("Content-Type", bodyWriter.FormDataContentType())
		pyClient := &http.Client{}
		pyResp, pyErr := pyClient.Do(pyReq)
		
		if pyErr == nil {
			defer pyResp.Body.Close()
			if pyResp.StatusCode == http.StatusOK {
				json.NewDecoder(pyResp.Body).Decode(&pyResponseBody)
			}
		}
	}

	// =========================================================================
	// 3. Call HF Spaces Whisper/Mistral (Assuming POST /transcribe logic exists)
	// =========================================================================
	var hfResponseBody interface{}
	
	// Reset file pointer again for HF Spaces
	file.Seek(0, 0)
	hfBuf := &bytes.Buffer{}
	hfWriter := multipart.NewWriter(hfBuf)
	
	hfWriter.WriteField("lyrics", lyrics)
	
	// Pass the aligned JSON from python directly if needed by HF spaces as a string
	if pyResponseBody != nil {
		alignBytes, _ := json.Marshal(pyResponseBody)
		hfWriter.WriteField("aligned_data", string(alignBytes))
	}
	
	hfFileWriter, err := hfWriter.CreateFormFile("file", header.Filename)
	if err == nil {
		io.Copy(hfFileWriter, file)
	}
	hfWriter.Close()
	
	// Note: Replace this URL with the actual Hugging Face Space URL in production
	hfReq, err := http.NewRequest("POST", "http://localhost:7860/transcribe", hfBuf)
	if err == nil {
		hfReq.Header.Set("Content-Type", hfWriter.FormDataContentType())
		hfClient := &http.Client{}
		hfResp, hfErr := hfClient.Do(hfReq)
		
		if hfErr == nil {
			defer hfResp.Body.Close()
			if hfResp.StatusCode == http.StatusOK {
				json.NewDecoder(hfResp.Body).Decode(&hfResponseBody)
			}
		} else {
			fmt.Println("Warning: HF Spaces call failed:", hfErr)
		}
	}

	// =========================================================================
	// 4. Return Final Combined Response
	// =========================================================================
	response := PipelineUploadResponse{
		AudioURL:       savedPath,
		FileSize:       header.Size,
		FileName:       header.Filename,
		Format:         filepath.Ext(header.Filename),
		AlignedLyrics:  pyResponseBody,
		MistralPayload: hfResponseBody,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}
