package storage

import (
	"io"
	"mime/multipart"
	"os"
	"path/filepath"
)

func SaveFile(file multipart.File, filename string) (string, error) {
	// Ensure uploads/audio directory exists
	dir := filepath.Join("uploads", "audio")
	err := os.MkdirAll(dir, 0755)
	if err != nil {
		return "", err
	}

	// Destination file
	dstPath := filepath.Join(dir, filename)
	out, err := os.Create(dstPath)
	if err != nil {
		return "", err
	}
	defer out.Close()

	// Copy file contents
	_, err = io.Copy(out, file)
	if err != nil {
		return "", err
	}

	// Web-accessible path format
	webUrl := "/uploads/audio/" + filename
	return webUrl, nil
}
