package com.raprank.backend.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Table(name = "tracks")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Track {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "artist_id", nullable = false)
    private Artist artist;

    @Column(nullable = false)
    private String title;

    @Column(columnDefinition = "TEXT", nullable = false)
    private String lyricsText;

    private String audioUrl;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private TrackStatus status; // PENDING, ANALYZED, FAILED

    @OneToOne(mappedBy = "track", cascade = CascadeType.ALL, orphanRemoval = true)
    private Score score;

    @OneToMany(mappedBy = "track", cascade = CascadeType.ALL, orphanRemoval = true)
    private java.util.List<Interaction> interactions;

    @Column(nullable = false)
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        status = TrackStatus.PENDING;
    }

    public enum TrackStatus {
        PENDING,
        DOWNLOADING_AUDIO,
        SEPARATING_AUDIO,
        TRANSCRIBING,
        ANALYZING_FLOW,
        ANALYZING_TEXT,
        ANALYZED,
        FAILED
    }
}
