package com.raprank.backend.repository;

import com.raprank.backend.entity.Score;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface ScoreRepository extends JpaRepository<Score, Long> {
    Optional<Score> findByTrackId(Long trackId);

    // Artist ka average score
    @Query("SELECT AVG(s.totalScore) FROM Score s WHERE s.track.artist.id = :artistId")
    Optional<Double> findAverageScoreByArtistId(@Param("artistId") Long artistId);
}
