package com.raprank.backend.service;

import com.raprank.backend.dto.request.LoginRequest;
import com.raprank.backend.dto.request.RegisterRequest;
import com.raprank.backend.dto.response.AuthResponse;
import com.raprank.backend.entity.Artist;
import com.raprank.backend.repository.ArtistRepository;
import com.raprank.backend.security.JwtService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;

import java.util.concurrent.CompletableFuture;

@Service
@RequiredArgsConstructor
public class AuthService {

    private final ArtistRepository artistRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;
    private final AuthenticationManager authenticationManager;
    private final RestTemplate restTemplate;

    @Value("${app.python.service.url}")
    private String pythonServiceUrl;

    private void wakeUpNlpServiceAsync() {
        CompletableFuture.runAsync(() -> {
            try {
                if (pythonServiceUrl != null && !pythonServiceUrl.isBlank()) {
                    String healthUrl = pythonServiceUrl.endsWith("/") 
                        ? pythonServiceUrl + "health" 
                        : pythonServiceUrl + "/health";
                    restTemplate.getForObject(healthUrl, String.class);
                }
            } catch (Exception ignored) {
                // Background warmup failure is silently ignored to prevent blocking authentication
            }
        });
    }

    @Transactional
    public AuthResponse register(RegisterRequest request) {
        if (artistRepository.existsByUsername(request.getUsername())) {
            throw new RuntimeException("Username already taken");
        }
        if (artistRepository.existsByEmail(request.getEmail())) {
            throw new RuntimeException("Email already registered");
        }

        Artist artist = Artist.builder()
                .username(request.getUsername())
                .email(request.getEmail())
                .passwordHash(passwordEncoder.encode(request.getPassword()))
                .bio(request.getBio())
                .badgeTitle("FRESH SPITTER")
                .build();

        artist = artistRepository.save(artist);

        UserDetails userDetails = User.builder()
                .username(artist.getUsername())
                .password(artist.getPasswordHash())
                .authorities("ROLE_ARTIST")
                .build();

        String token = jwtService.generateToken(userDetails);
        wakeUpNlpServiceAsync();

        return AuthResponse.builder()
                .token(token)
                .artistId(artist.getId())
                .username(artist.getUsername())
                .email(artist.getEmail())
                .badgeTitle(artist.getBadgeTitle())
                .build();
    }

    public AuthResponse login(LoginRequest request) {
        authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(request.getUsername(), request.getPassword())
        );

        Artist artist = artistRepository.findByUsername(request.getUsername())
                .orElseThrow(() -> new RuntimeException("Artist not found"));

        UserDetails userDetails = User.builder()
                .username(artist.getUsername())
                .password(artist.getPasswordHash())
                .authorities("ROLE_ARTIST")
                .build();

        String token = jwtService.generateToken(userDetails);
        wakeUpNlpServiceAsync();

        return AuthResponse.builder()
                .token(token)
                .artistId(artist.getId())
                .username(artist.getUsername())
                .email(artist.getEmail())
                .badgeTitle(artist.getBadgeTitle())
                .build();
    }
}
