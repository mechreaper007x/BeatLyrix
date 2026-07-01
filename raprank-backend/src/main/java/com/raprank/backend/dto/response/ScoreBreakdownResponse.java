package com.raprank.backend.dto.response;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonAlias;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@JsonIgnoreProperties(ignoreUnknown = true)
public class ScoreBreakdownResponse {
    @JsonAlias("syllable_score")
    private Double syllableScore;

    @JsonAlias("alliteration_score")
    private Double alliterationScore;

    @JsonAlias("flow_score")
    private Double flowScore;

    @JsonAlias("total_score")
    private Double totalScore;

    private String grade;

    @JsonAlias("rhyme_score")
    private Double rhymeScore;

    @JsonAlias("wordplay_score")
    private Double wordplayScore;

    @JsonAlias("syllable_weight")
    private Double syllableWeight;

    @JsonAlias("vocabulary_uniqueness")
    private Double vocabularyUniqueness;

    @JsonAlias("double_entendres_count")
    private Integer doubleEntendresCount;

    @JsonAlias("puns_count")
    private Integer punsCount;

    @JsonAlias("similes_count")
    private Integer similesCount;

    @JsonAlias("metaphors_count")
    private Integer metaphorsCount;
}
