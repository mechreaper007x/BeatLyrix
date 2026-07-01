package com.raprank.backend.repository;

import com.raprank.backend.entity.Track;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface TrackRepository extends JpaRepository<Track, Long> {
    List<Track> findByArtistIdOrderByCreatedAtDesc(Long artistId);
    List<Track> findByStatus(Track.TrackStatus status);

    // Leaderboard query — tracks with scores, sorted by totalScore desc
    @Query("""
        SELECT t FROM Track t
        JOIN Score s ON s.track.id = t.id
        WHERE t.status = 'ANALYZED'
        ORDER BY s.totalScore DESC
        """)
    List<Track> findTopRankedTracks(Pageable pageable);
}
