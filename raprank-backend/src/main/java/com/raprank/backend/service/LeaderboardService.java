package com.raprank.backend.service;

import com.raprank.backend.dto.response.LeaderboardEntryResponse;
import com.raprank.backend.entity.Artist;
import com.raprank.backend.entity.Track;
import com.raprank.backend.repository.ArtistRepository;
import com.raprank.backend.repository.ScoreRepository;
import com.raprank.backend.repository.TrackRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class LeaderboardService {

    private final TrackRepository trackRepository;
    private final ScoreRepository scoreRepository;
    private final ArtistRepository artistRepository;
    private final RedisTemplate<String, Object> redisTemplate;

    private static final String CACHE_KEY_PREFIX = "leaderboard:page:";

    @Transactional
    public List<LeaderboardEntryResponse> getLeaderboard(int page, int size, String currentUsername) {
        String cacheKey = CACHE_KEY_PREFIX + page;

        // Try to fetch from Redis Cache
        try {
            List<?> cachedList = (List<?>) redisTemplate.opsForValue().get(cacheKey);
            if (cachedList != null) {
                // Safely cast and map current user flags
                List<LeaderboardEntryResponse> entries = cachedList.stream()
                        .filter(item -> item instanceof LeaderboardEntryResponse)
                        .map(item -> (LeaderboardEntryResponse) item)
                        .collect(Collectors.toList());
                return updateCurrentUserFlag(entries, currentUsername);
            }
        } catch (Exception e) {
            // Fallback silently to DB on Redis exceptions
        }

        // Cache miss -> fetch from DB
        Pageable pageable = PageRequest.of(page, size);
        List<Track> topTracks = trackRepository.findTopRankedTracks(pageable);

        List<LeaderboardEntryResponse> entries = new ArrayList<>();
        int rank = page * size + 1;

        for (Track track : topTracks) {
            Artist artist = track.getArtist();
            double avgScore = scoreRepository.findAverageScoreByArtistId(artist.getId()).orElse(0.0);
            int trackCount = trackRepository.findByArtistIdOrderByCreatedAtDesc(artist.getId()).size();

            String artistGrade = getGrade(avgScore);
            String badge = getBadgeTitle(avgScore);

            // Update badge title in DB if changed
            if (!badge.equals(artist.getBadgeTitle())) {
                artist.setBadgeTitle(badge);
                artistRepository.save(artist);
            }

            LeaderboardEntryResponse entry = LeaderboardEntryResponse.builder()
                    .rank(rank++)
                    .artistId(artist.getId())
                    .artistUsername(artist.getUsername())
                    .artistBadge(badge)
                    .trackCount(trackCount)
                    .avgScore(avgScore)
                    .grade(artistGrade)
                    .isCurrentUser(artist.getUsername().equals(currentUsername))
                    .build();

            entries.add(entry);
        }

        // Cache the result for 5 minutes
        try {
            redisTemplate.opsForValue().set(cacheKey, entries, Duration.ofMinutes(5));
        } catch (Exception e) {
            // Log/ignore Redis write error
        }

        return entries;
    }

    private List<LeaderboardEntryResponse> updateCurrentUserFlag(List<LeaderboardEntryResponse> list, String currentUsername) {
        return list.stream()
                .map(entry -> LeaderboardEntryResponse.builder()
                        .rank(entry.getRank())
                        .artistId(entry.getArtistId())
                        .artistUsername(entry.getArtistUsername())
                        .artistBadge(entry.getArtistBadge())
                        .trackCount(entry.getTrackCount())
                        .avgScore(entry.getAvgScore())
                        .grade(entry.getGrade())
                        .isCurrentUser(entry.getArtistUsername().equals(currentUsername))
                        .build())
                .collect(Collectors.toList());
    }

    public String getGrade(Double score) {
        if (score == null) return "D";
        if (score >= 90) return "S";
        if (score >= 80) return "A";
        if (score >= 70) return "B";
        if (score >= 60) return "C";
        return "D";
    }

    public String getBadgeTitle(Double avgScore) {
        if (avgScore == null) return "FRESH SPITTER";
        if (avgScore >= 90) return "PLATINUM SPITTER";
        if (avgScore >= 80) return "GOLD SPITTER";
        if (avgScore >= 70) return "SILVER SPITTER";
        return "FRESH SPITTER";
    }
}
