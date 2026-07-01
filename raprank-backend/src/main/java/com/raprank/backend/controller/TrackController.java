package com.raprank.backend.controller;

import com.raprank.backend.dto.request.TrackUploadRequest;
import com.raprank.backend.dto.response.TrackResponse;
import com.raprank.backend.repository.ArtistRepository;
import com.raprank.backend.service.TrackService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/tracks")
@RequiredArgsConstructor
public class TrackController {

    private final TrackService trackService;
    private final ArtistRepository artistRepository;

    @PostMapping
    public ResponseEntity<TrackResponse> uploadTrack(
            @RequestBody @Valid TrackUploadRequest request,
            @AuthenticationPrincipal UserDetails userDetails
    ) {
        Long artistId = getArtistId(userDetails);
        return ResponseEntity.status(HttpStatus.CREATED).body(trackService.uploadTrack(request, artistId));
    }

    @GetMapping("/{id}")
    public ResponseEntity<TrackResponse> getTrackById(@PathVariable Long id) {
        return ResponseEntity.ok(trackService.getTrackById(id));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteTrack(
            @PathVariable Long id,
            @AuthenticationPrincipal UserDetails userDetails
    ) {
        Long artistId = getArtistId(userDetails);
        trackService.deleteTrack(id, artistId);
        return ResponseEntity.noContent().build();
    }

    @PostMapping("/{id}/status")
    public ResponseEntity<Void> updateTrackStatus(
            @PathVariable Long id,
            @RequestParam com.raprank.backend.entity.Track.TrackStatus status
    ) {
        trackService.updateTrackStatus(id, status);
        return ResponseEntity.ok().build();
    }

    private Long getArtistId(UserDetails userDetails) {
        return artistRepository.findByUsername(userDetails.getUsername())
                .orElseThrow(() -> new RuntimeException("Artist not found"))
                .getId();
    }
}
