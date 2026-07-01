package com.raprank.backend.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class LeaderboardEntryResponse {
    private Integer rank;
    private Long artistId;
    private String artistUsername;
    private String artistBadge;
    private int trackCount;
    private Double avgScore;
    private String grade;
    private boolean isCurrentUser; // logged in user ka row highlight karne ke liye
}
