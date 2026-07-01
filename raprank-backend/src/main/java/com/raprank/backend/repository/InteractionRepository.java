package com.raprank.backend.repository;

import com.raprank.backend.entity.Interaction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface InteractionRepository extends JpaRepository<Interaction, Long> {

    // Like count per track
    long countByTrackIdAndType(Long trackId, Interaction.InteractionType type);

    // Check if artist already liked a track
    boolean existsByTrackIdAndArtistIdAndType(
        Long trackId, Long artistId, Interaction.InteractionType type
    );

    // Comments for a track
    List<Interaction> findByTrackIdAndTypeOrderByCreatedAtDesc(
        Long trackId, Interaction.InteractionType type
    );

    // Artist ka like remove karna
    void deleteByTrackIdAndArtistIdAndType(
        Long trackId, Long artistId, Interaction.InteractionType type
    );
}
