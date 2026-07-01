package com.raprank.backend.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TrackResponse {
    private Long id;
    private String title;
    private String lyricsText;
    private String audioUrl;
    private String artistUsername;
    private Long artistId;
    private Double totalScore;
    private String grade;          // S, A, B, C, D — computed from totalScore
    private long likeCount;
    private long commentCount;
    private LocalDateTime createdAt;
    private ScoreBreakdownResponse scoreBreakdown;
}
