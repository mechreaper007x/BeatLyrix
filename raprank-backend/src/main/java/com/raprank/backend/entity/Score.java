package com.raprank.backend.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Table(name = "scores")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Score {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "track_id", nullable = false, unique = true)
    private Track track;

    @Column(nullable = false)
    private Double syllableScore;   // 0-100

    @Column(nullable = true)
    private Double flowScore;       // 0-100 (nullable if audio analysis fails or is missing)

    @Column(nullable = false)
    private Double totalScore;      // weighted average

    @Column(columnDefinition = "TEXT")
    private String breakdownJson;   // full JSON from Python service

    private LocalDateTime analyzedAt;

    @PrePersist
    protected void onAnalyze() {
        analyzedAt = LocalDateTime.now();
    }
}
