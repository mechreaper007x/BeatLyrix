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

    @JsonAlias("vocabulary_score")
    private Double vocabularyScore;

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

    @JsonAlias("assonance_score")
    private Double assonanceScore;

    @JsonAlias("consonance_score")
    private Double consonanceScore;

    @JsonAlias("onomatopoeia_score")
    private Double onomatopoeiaScore;

    @JsonAlias("codeswitch_score")
    private Double codeswitchScore;

    @JsonAlias("repetition_score")
    private Double repetitionScore;

    @JsonAlias("cadence_text_score")
    private Double cadenceTextScore;

    @JsonAlias("callback_score")
    private Double callbackScore;

    @JsonAlias("punchline_count")
    private Integer punchlineCount;

    @JsonAlias("extended_metaphor_count")
    private Integer extendedMetaphorCount;

    @JsonAlias("allusions_count")
    private Integer allusionsCount;

    @JsonAlias("multisyllabic_rhyme_count")
    private Integer multisyllabicRhymeCount;

    @JsonAlias("generated_lyrics")
    private String generatedLyrics;

    @JsonAlias("wordplay_explanation")
    private String wordplayExplanation;

    @JsonAlias("nlp_explanations")
    private java.util.Map<String, String> nlpExplanations;

    @JsonAlias("style_cluster")
    private String styleCluster;

    @JsonAlias("style_cluster_confidence")
    private Double styleClusterConfidence;

    @JsonAlias("style_membership")
    private java.util.Map<String, Double> styleMembership;

    @JsonAlias("element_clusters")
    private java.util.Map<String, Object> elementClusters;

    @JsonAlias("predicted_tier")
    private String predictedTier;

    @JsonAlias("tier_confidence")
    private Double tierConfidence;

    @JsonAlias("tier_probabilities")
    private java.util.Map<String, Double> tierProbabilities;

    @JsonAlias("svm_tier")
    private String svmTier;

    @JsonAlias("svm_tier_confidence")
    private Double svmTierConfidence;

    @JsonAlias("svm_tier_probabilities")
    private java.util.Map<String, Double> svmTierProbabilities;

    @JsonAlias("bayes_tier")
    private String bayesTier;

    @JsonAlias("bayes_tier_probabilities")
    private java.util.Map<String, Double> bayesTierProbabilities;

    @JsonAlias("tier_consensus")
    private String tierConsensus;

    @JsonAlias("tier_consensus_agreement")
    private Double tierConsensusAgreement;
}
