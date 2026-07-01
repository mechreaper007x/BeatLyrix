package com.raprank.backend.repository;

import com.raprank.backend.entity.Artist;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface ArtistRepository extends JpaRepository<Artist, Long> {
    Optional<Artist> findByUsername(String username);
    Optional<Artist> findByEmail(String email);
    boolean existsByUsername(String username);
    boolean existsByEmail(String email);
}
