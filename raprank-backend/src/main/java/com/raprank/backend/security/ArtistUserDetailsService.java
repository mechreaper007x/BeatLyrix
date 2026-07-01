package com.raprank.backend.security;

import com.raprank.backend.repository.ArtistRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class ArtistUserDetailsService implements UserDetailsService {

    private final ArtistRepository artistRepository;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        return artistRepository.findByUsername(username)
                .map(artist -> User.builder()
                        .username(artist.getUsername())
                        .password(artist.getPasswordHash())
                        .authorities("ROLE_ARTIST")
                        .build())
                .orElseThrow(() -> new UsernameNotFoundException("Artist not found with username: " + username));
    }
}
