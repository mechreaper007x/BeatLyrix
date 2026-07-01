package com.raprank.backend.service;

import com.raprank.backend.dto.response.ArtistProfileResponse;
import com.raprank.backend.dto.response.TrackResponse;
import com.raprank.backend.entity.Artist;
import com.raprank.backend.entity.Track;
import com.raprank.backend.repository.ArtistRepository;
import com.raprank.backend.repository.ScoreRepository;
import com.raprank.backend.repository.TrackRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ArtistService {

    private final ArtistRepository artistRepository;
    private final TrackRepository trackRepository;
    private final ScoreRepository scoreRepository;
    private final TrackService trackService;

    public ArtistProfileResponse getArtistProfile(Long artistId, String currentUsername) {
        Artist artist = artistRepository.findById(artistId)
                .orElseThrow(() -> new RuntimeException("Artist not found"));

        List<TrackResponse> tracks = trackRepository.findByArtistIdOrderByCreatedAtDesc(artistId).stream()
                .map(trackService::mapToTrackResponse)
                .collect(Collectors.toList());

        double avgScore = scoreRepository.findAverageScoreByArtistId(artistId).orElse(0.0);
        int rank = calculateArtistRank(artistId);
        String badge = getBadgeTitle(avgScore);

        // Update badge title in DB if changed
        if (!badge.equals(artist.getBadgeTitle())) {
            artist.setBadgeTitle(badge);
            artistRepository.save(artist);
        }

        return ArtistProfileResponse.builder()
                .id(artist.getId())
                .username(artist.getUsername())
                .bio(artist.getBio())
                .profilePicUrl(artist.getProfilePicUrl())
                .badgeTitle(badge)
                .rank(rank > 0 ? rank : null)
                .avgScore(avgScore)
                .grade(getGrade(avgScore))
                .totalTracks(tracks.size())
                .tracks(tracks)
                .createdAt(artist.getCreatedAt())
                .build();
    }

    public ArtistProfileResponse getCurrentArtist(String username) {
        Artist artist = artistRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("Artist not found"));
        return getArtistProfile(artist.getId(), username);
    }

    @Transactional
    public ArtistProfileResponse updateProfile(Long artistId, String bio, String profilePicUrl) {
        Artist artist = artistRepository.findById(artistId)
                .orElseThrow(() -> new RuntimeException("Artist not found"));

        if (bio != null) {
            artist.setBio(bio);
        }
        if (profilePicUrl != null) {
            artist.setProfilePicUrl(profilePicUrl);
        }

        artist = artistRepository.save(artist);
        return getArtistProfile(artist.getId(), artist.getUsername());
    }

    private int calculateArtistRank(Long artistId) {
        List<Object[]> artistAvgScores = trackRepository.findAll().stream()
                .map(Track::getArtist)
                .distinct()
                .map(artist -> new Object[]{
                        artist.getId(),
                        scoreRepository.findAverageScoreByArtistId(artist.getId()).orElse(0.0)
                })
                .sorted((a, b) -> Double.compare((Double) b[1], (Double) a[1]))
                .collect(Collectors.toList());

        for (int i = 0; i < artistAvgScores.size(); i++) {
            if (artistAvgScores.get(i)[0].equals(artistId)) {
                return i + 1;
            }
        }
        return 0;
    }

    private String getBadgeTitle(Double avgScore) {
        if (avgScore == null) return "FRESH SPITTER";
        if (avgScore >= 90) return "PLATINUM SPITTER";
        if (avgScore >= 80) return "GOLD SPITTER";
        if (avgScore >= 70) return "SILVER SPITTER";
        return "FRESH SPITTER";
    }

    private String getGrade(Double score) {
        if (score == null) return "D";
        if (score >= 90) return "S";
        if (score >= 80) return "A";
        if (score >= 70) return "B";
        if (score >= 60) return "C";
        return "D";
    }
}
