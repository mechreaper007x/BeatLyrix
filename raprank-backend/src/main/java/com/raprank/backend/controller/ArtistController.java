package com.raprank.backend.controller;

import com.raprank.backend.dto.response.ArtistProfileResponse;
import com.raprank.backend.dto.response.TrackResponse;
import com.raprank.backend.service.ArtistService;
import com.raprank.backend.service.TrackService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/artists")
@RequiredArgsConstructor
public class ArtistController {

    private final ArtistService artistService;
    private final TrackService trackService;

    @GetMapping("/me")
    public ResponseEntity<ArtistProfileResponse> getCurrentArtist(
            @AuthenticationPrincipal UserDetails userDetails
    ) {
        return ResponseEntity.ok(artistService.getCurrentArtist(userDetails.getUsername()));
    }

    @GetMapping("/{id}")
    public ResponseEntity<ArtistProfileResponse> getArtistProfile(
            @PathVariable Long id,
            @AuthenticationPrincipal UserDetails userDetails
    ) {
        String currentUsername = userDetails != null ? userDetails.getUsername() : null;
        return ResponseEntity.ok(artistService.getArtistProfile(id, currentUsername));
    }

    @GetMapping("/{id}/tracks")
    public ResponseEntity<List<TrackResponse>> getTracksByArtist(@PathVariable Long id) {
        return ResponseEntity.ok(trackService.getTracksByArtist(id));
    }
}
