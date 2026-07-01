package com.raprank.backend.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ScoreBreakdownResponse {
    private Double syllableScore;
    private Double alliterationScore;
    private Double flowScore;
    private Double totalScore;
    private String grade;
}
