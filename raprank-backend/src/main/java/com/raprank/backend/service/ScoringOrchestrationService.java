package com.raprank.backend.service;

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

    @Value("${app.python.service.url}")
    private String pythonServiceUrl;

    @Value("${app.go.service.url}")
    private String goServiceUrl;

    @Async
    @Transactional
    public void triggerAnalysis(Track track) {
        try {
            String url = pythonServiceUrl + "/analyze";
            Map<String, String> request = new HashMap<>();
            request.put("lyrics", track.getLyricsText());
            if (track.getAudioUrl() != null && !track.getAudioUrl().isEmpty()) {
                request.put("audio_url", goServiceUrl + track.getAudioUrl());
            }

            log.info("Sending track {} to NLP service at {} (audio_url: {})...", 
                    track.getId(), url, request.get("audio_url"));
            PyAnalysisResponse response = restTemplate.postForObject(url, request, PyAnalysisResponse.class);

            if (response != null) {
                String jsonBreakdown = objectMapper.writeValueAsString(response);

                Score score = Score.builder()
                        .track(track)
                        .syllableScore(response.getSyllable_score())
                        .alliterationScore(response.getAlliteration_score())
                        .flowScore(response.getFlow_score())
                        .totalScore(response.getTotal_score())
                        .breakdownJson(jsonBreakdown)
                        .build();

                scoreRepository.save(score);

                track.setStatus(Track.TrackStatus.ANALYZED);
                trackRepository.save(track);
                log.info("Track {} successfully scored and analyzed.", track.getId());
            } else {
                throw new RuntimeException("Received empty response from NLP service");
            }
        } catch (Exception e) {
            log.error("Failed to analyze track " + track.getId(), e);
            track.setStatus(Track.TrackStatus.FAILED);
            trackRepository.save(track);
        }
    }

    @Data
    public static class PyAnalysisResponse {
        private Double syllable_score;
        private Double alliteration_score;
        private Double flow_score;
        private Double total_score;
        private Integer word_count;
        private Integer line_count;
        private Double avg_syllables_per_word;
        private Object alliteration_pairs;
    }
}
