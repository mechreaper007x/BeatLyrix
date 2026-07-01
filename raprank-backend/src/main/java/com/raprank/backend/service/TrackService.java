package com.raprank.backend.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.raprank.backend.dto.request.TrackUploadRequest;
import com.raprank.backend.dto.response.ScoreBreakdownResponse;
import com.raprank.backend.dto.response.TrackResponse;
import com.raprank.backend.entity.Artist;
import com.raprank.backend.entity.Interaction;
import com.raprank.backend.entity.Score;
import com.raprank.backend.entity.Track;
import com.raprank.backend.repository.ArtistRepository;
import com.raprank.backend.repository.InteractionRepository;
import com.raprank.backend.repository.ScoreRepository;
import com.raprank.backend.repository.TrackRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class TrackService {

    private final TrackRepository trackRepository;
    private final ArtistRepository artistRepository;
    private final ScoreRepository scoreRepository;
    private final InteractionRepository interactionRepository;
    private final ScoringOrchestrationService scoringOrchestrationService;
    private final ObjectMapper objectMapper;

    @Transactional
    public TrackResponse uploadTrack(TrackUploadRequest request, Long artistId) {
        Artist artist = artistRepository.findById(artistId)
                .orElseThrow(() -> new RuntimeException("Artist not found"));

        Track track = Track.builder()
                .title(request.getTitle())
                .lyricsText(request.getLyricsText())
                .audioUrl(request.getAudioUrl())
                .artist(artist)
                .status(Track.TrackStatus.PENDING)
                .build();

        track = trackRepository.save(track);

        // Async orchestration call
        scoringOrchestrationService.triggerAnalysis(track);

        return mapToTrackResponse(track);
    }

    public TrackResponse getTrackById(Long trackId) {
        Track track = trackRepository.findById(trackId)
                .orElseThrow(() -> new RuntimeException("Track not found"));
        return mapToTrackResponse(track);
    }

    public List<TrackResponse> getTracksByArtist(Long artistId) {
        return trackRepository.findByArtistIdOrderByCreatedAtDesc(artistId).stream()
                .map(this::mapToTrackResponse)
                .collect(Collectors.toList());
    }

    @Transactional
    public void deleteTrack(Long trackId, Long requestingArtistId) {
        Track track = trackRepository.findById(trackId)
                .orElseThrow(() -> new RuntimeException("Track not found"));

        if (!track.getArtist().getId().equals(requestingArtistId)) {
            throw new RuntimeException("Not authorized to delete this track");
        }

        trackRepository.delete(track);
    }

    @Transactional
    public void updateTrackStatus(Long trackId, Track.TrackStatus status) {
        Track track = trackRepository.findById(trackId)
                .orElseThrow(() -> new RuntimeException("Track not found"));
        track.setStatus(status);
        trackRepository.save(track);
    }

    public TrackResponse mapToTrackResponse(Track track) {
        long likeCount = interactionRepository.countByTrackIdAndType(track.getId(), Interaction.InteractionType.LIKE);
        long commentCount = interactionRepository.countByTrackIdAndType(track.getId(), Interaction.InteractionType.COMMENT);

        ScoreBreakdownResponse breakdown = null;
        Double totalScore = null;
        String grade = null;

        var scoreOpt = scoreRepository.findByTrackId(track.getId());
        if (scoreOpt.isPresent()) {
            Score score = scoreOpt.get();
            totalScore = score.getTotalScore();
            grade = getGrade(totalScore);
            if (score.getBreakdownJson() != null && !score.getBreakdownJson().isEmpty()) {
                try {
                    breakdown = objectMapper.readValue(score.getBreakdownJson(), ScoreBreakdownResponse.class);
                    breakdown.setTotalScore(totalScore);
                    breakdown.setGrade(grade);
                } catch (Exception e) {
                    breakdown = ScoreBreakdownResponse.builder()
                            .syllableScore(score.getSyllableScore())
                            .alliterationScore(score.getAlliterationScore())
                            .flowScore(score.getFlowScore())
                            .totalScore(totalScore)
                            .grade(grade)
                            .build();
                }
            } else {
                breakdown = ScoreBreakdownResponse.builder()
                        .syllableScore(score.getSyllableScore())
                        .alliterationScore(score.getAlliterationScore())
                        .flowScore(score.getFlowScore())
                        .totalScore(totalScore)
                        .grade(grade)
                        .build();
            }
        }

        return TrackResponse.builder()
                .id(track.getId())
                .title(track.getTitle())
                .lyricsText(track.getLyricsText())
                .audioUrl(track.getAudioUrl())
                .artistUsername(track.getArtist().getUsername())
                .artistId(track.getArtist().getId())
                .totalScore(totalScore)
                .grade(grade)
                .likeCount(likeCount)
                .commentCount(commentCount)
                .createdAt(track.getCreatedAt())
                .status(track.getStatus().name())
                .scoreBreakdown(breakdown)
                .build();
    }

    private String getGrade(Double score) {
        if (score == null) return null;
        if (score >= 90) return "S";
        if (score >= 80) return "A";
        if (score >= 70) return "B";
        if (score >= 60) return "C";
        return "D";
    }
}
