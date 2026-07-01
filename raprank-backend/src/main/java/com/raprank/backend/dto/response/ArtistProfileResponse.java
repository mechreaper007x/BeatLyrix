package com.raprank.backend.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ArtistProfileResponse {
    private Long id;
    private String username;
    private String bio;
    private String profilePicUrl;
    private String badgeTitle;
    private Integer rank;
    private Double avgScore;
    private String grade;
    private int totalTracks;
    private List<TrackResponse> tracks;
    private LocalDateTime createdAt;
}
