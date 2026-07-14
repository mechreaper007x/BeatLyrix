package com.raprank.backend.service;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.raprank.backend.entity.Score;
import com.raprank.backend.entity.Track;
import com.raprank.backend.repository.ScoreRepository;
import com.raprank.backend.repository.TrackRepository;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;

import org.springframework.transaction.support.TransactionTemplate;

import java.util.HashMap;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class ScoringOrchestrationService {

    private final RestTemplate restTemplate;
    private final TrackRepository trackRepository;
    private final ScoreRepository scoreRepository;
    private final ObjectMapper objectMapper;
    private final TransactionTemplate transactionTemplate;

    @Value("${app.python.service.url}")
    private String pythonServiceUrl;

    @Value("${app.go.service.url}")
    private String goServiceUrl;

    @Async
    public void triggerAnalysis(Track track) {
        try {
            String url = pythonServiceUrl + "/analyze";
            Map<String, Object> request = new HashMap<>();
            request.put("lyrics", track.getLyricsText());
            request.put("track_id", track.getId());
            if (track.getAudioUrl() != null && !track.getAudioUrl().isEmpty()) {
                request.put("audio_url", goServiceUrl + track.getAudioUrl());
            }

            log.info("Sending track {} to NLP service at {} (audio_url: {})...", 
                    track.getId(), url, request.get("audio_url"));
            PyAnalysisResponse response = restTemplate.postForObject(url, request, PyAnalysisResponse.class);

            if (response != null) {
                final PyAnalysisResponse finalResponse = response;
                transactionTemplate.executeWithoutResult(status -> {
                    Track currentTrack = trackRepository.findById(track.getId())
                            .orElseThrow(() -> new RuntimeException("Track not found: " + track.getId()));
                    try {
                        String jsonBreakdown = objectMapper.writeValueAsString(finalResponse);

                        Score score = Score.builder()
                                .track(currentTrack)
                                .syllableScore(finalResponse.getSyllable_score())
                                .flowScore(finalResponse.getFlow_score())
                                .totalScore(finalResponse.getTotal_score())
                                .breakdownJson(jsonBreakdown)
                                .build();

                        scoreRepository.save(score);

                        if (finalResponse.getGenerated_lyrics() != null) {
                            currentTrack.setLyricsText(finalResponse.getGenerated_lyrics());
                        }
                        
                        currentTrack.setStatus(Track.TrackStatus.ANALYZED);
                        trackRepository.save(currentTrack);
                        log.info("Track {} successfully scored and analyzed.", currentTrack.getId());
                    } catch (Exception e) {
                        throw new RuntimeException("Failed to save analysis score for track " + currentTrack.getId(), e);
                    }
                });
            } else {
                throw new RuntimeException("Received empty response from NLP service");
            }
        } catch (Exception e) {
            log.error("Failed to analyze track " + track.getId(), e);
            try {
                transactionTemplate.executeWithoutResult(status -> {
                    trackRepository.findById(track.getId()).ifPresent(currentTrack -> {
                        currentTrack.setStatus(Track.TrackStatus.FAILED);
                        trackRepository.save(currentTrack);
                    });
                });
            } catch (Exception ex) {
                log.error("Failed to set track status to FAILED", ex);
            }
        }
    }

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class PyAnalysisResponse {
        private Double syllable_score;
        private Double flow_score;
        private Double total_score;
        private Integer word_count;
        private Integer line_count;
        private Double avg_syllables_per_word;

        private Double rhyme_score;
        private Double wordplay_score;
        private Double syllable_weight;
        private Double vocabulary_score;
        private Double vocabulary_uniqueness;
        private Integer double_entendres_count;
        private Integer puns_count;
        private Integer similes_count;
        private Integer metaphors_count;

        private Double assonance_score;
        private Double consonance_score;
        private Double onomatopoeia_score;
        private Double codeswitch_score;
        private Double repetition_score;
        private Double cadence_text_score;
        private Double callback_score;
        private Integer punchline_count;
        private Integer extended_metaphor_count;
        private Integer allusions_count;
        private Integer multisyllabic_rhyme_count;

        private String generated_lyrics;
        private String wordplay_explanation;
        private java.util.Map<String, String> nlp_explanations;

        private String style_cluster;
        private Double style_cluster_confidence;
        private java.util.Map<String, Double> style_membership;

        private java.util.Map<String, Object> element_clusters;

        private String predicted_tier;
        private Double tier_confidence;
        private java.util.Map<String, Double> tier_probabilities;
    }
}
