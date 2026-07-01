package com.raprank.backend.dto.request;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class TrackUploadRequest {
    @NotBlank
    @Size(max = 100)
    private String title;

    @NotBlank
    private String lyricsText;

    private String audioUrl; // from Go upload service
}
