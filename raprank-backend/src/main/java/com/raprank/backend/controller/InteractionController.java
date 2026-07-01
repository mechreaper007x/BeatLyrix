package com.raprank.backend.controller;

import com.raprank.backend.dto.request.CommentRequest;
import com.raprank.backend.dto.response.CommentResponse;
import com.raprank.backend.entity.Interaction;
import com.raprank.backend.repository.ArtistRepository;
import com.raprank.backend.repository.InteractionRepository;
import com.raprank.backend.service.InteractionService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/tracks/{trackId}")
@RequiredArgsConstructor
public class InteractionController {

    private final InteractionService interactionService;
    private final ArtistRepository artistRepository;
    private final InteractionRepository interactionRepository;

    @PostMapping("/like")
    public ResponseEntity<Map<String, Object>> toggleLike(
            @PathVariable Long trackId,
            @AuthenticationPrincipal UserDetails userDetails
    ) {
        Long artistId = getArtistId(userDetails);
        boolean wasLiked = interactionRepository.existsByTrackIdAndArtistIdAndType(
                trackId, artistId, Interaction.InteractionType.LIKE
        );
        long count = interactionService.toggleLike(trackId, artistId);

        Map<String, Object> response = new HashMap<>();
        response.put("liked", !wasLiked);
        response.put("likeCount", count);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/comments")
    public ResponseEntity<CommentResponse> addComment(
            @PathVariable Long trackId,
            @RequestBody @Valid CommentRequest request,
            @AuthenticationPrincipal UserDetails userDetails
    ) {
        Long artistId = getArtistId(userDetails);
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(interactionService.addComment(trackId, artistId, request));
    }

    @GetMapping("/comments")
    public ResponseEntity<List<CommentResponse>> getComments(@PathVariable Long trackId) {
        return ResponseEntity.ok(interactionService.getComments(trackId));
    }

    private Long getArtistId(UserDetails userDetails) {
        return artistRepository.findByUsername(userDetails.getUsername())
                .orElseThrow(() -> new RuntimeException("Artist not found"))
                .getId();
    }
}
