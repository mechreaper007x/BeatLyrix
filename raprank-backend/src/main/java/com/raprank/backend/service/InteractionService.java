package com.raprank.backend.service;

import com.raprank.backend.dto.request.CommentRequest;
import com.raprank.backend.dto.response.CommentResponse;
import com.raprank.backend.entity.Artist;
import com.raprank.backend.entity.Interaction;
import com.raprank.backend.entity.Track;
import com.raprank.backend.repository.ArtistRepository;
import com.raprank.backend.repository.InteractionRepository;
import com.raprank.backend.repository.TrackRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class InteractionService {

    private final InteractionRepository interactionRepository;
    private final TrackRepository trackRepository;
    private final ArtistRepository artistRepository;

    @Transactional
    public long toggleLike(Long trackId, Long artistId) {
        Track track = trackRepository.findById(trackId)
                .orElseThrow(() -> new RuntimeException("Track not found"));
        Artist artist = artistRepository.findById(artistId)
                .orElseThrow(() -> new RuntimeException("Artist not found"));

        boolean alreadyLiked = interactionRepository.existsByTrackIdAndArtistIdAndType(
                trackId, artistId, Interaction.InteractionType.LIKE
        );

        if (alreadyLiked) {
            interactionRepository.deleteByTrackIdAndArtistIdAndType(
                    trackId, artistId, Interaction.InteractionType.LIKE
            );
        } else {
            Interaction like = Interaction.builder()
                    .track(track)
                    .artist(artist)
                    .type(Interaction.InteractionType.LIKE)
                    .build();
            interactionRepository.save(like);
        }

        return interactionRepository.countByTrackIdAndType(trackId, Interaction.InteractionType.LIKE);
    }

    @Transactional
    public CommentResponse addComment(Long trackId, Long artistId, CommentRequest request) {
        Track track = trackRepository.findById(trackId)
                .orElseThrow(() -> new RuntimeException("Track not found"));
        Artist artist = artistRepository.findById(artistId)
                .orElseThrow(() -> new RuntimeException("Artist not found"));

        Interaction comment = Interaction.builder()
                .track(track)
                .artist(artist)
                .type(Interaction.InteractionType.COMMENT)
                .content(request.getContent())
                .build();

        comment = interactionRepository.save(comment);

        return CommentResponse.builder()
                .id(comment.getId())
                .artistUsername(artist.getUsername())
                .content(comment.getContent())
                .createdAt(comment.getCreatedAt())
                .build();
    }

    public List<CommentResponse> getComments(Long trackId) {
        return interactionRepository.findByTrackIdAndTypeOrderByCreatedAtDesc(
                trackId, Interaction.InteractionType.COMMENT
        ).stream()
                .map(comment -> CommentResponse.builder()
                        .id(comment.getId())
                        .artistUsername(comment.getArtist().getUsername())
                        .content(comment.getContent())
                        .createdAt(comment.getCreatedAt())
                        .build())
                .collect(Collectors.toList());
    }
}
