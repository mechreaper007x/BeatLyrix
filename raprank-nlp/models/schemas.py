"""
Pydantic schemas for RapRank NLP Service v2.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class WordTimestamp(BaseModel):
    """Word-level timing entry from faster-whisper."""
    word: str
    start: float
    end: float


class AnalyzeRequest(BaseModel):
    lyrics: str = Field(..., description="Full lyric text to analyse.")
    language: str = Field(
        default="auto",
        description="ISO language hint: 'en', 'hi', or 'auto' for auto-detect.",
    )
    words: Optional[List[WordTimestamp]] = Field(
        default=None,
        description=(
            "Word-level timestamps from faster-whisper. "
            "When provided, enables flow/beat-sync scoring without audio."
        ),
    )
    audio_url: Optional[str] = Field(
        default=None,
        description="Absolute URL to download the audio track for flow scoring.",
    )
    track_id: Optional[int] = Field(
        default=None,
        description="Optional core registry database track ID for real-time status updates.",
    )


# ── Sub-models ────────────────────────────────────────────────────────────────

class RhymeMatch(BaseModel):
    """A detected rhyme pair between two lyric lines."""
    line_a: int = Field(..., description="0-based index of the first line.")
    line_b: int = Field(..., description="0-based index of the second line.")
    word_a: str
    word_b: str
    is_multisyllabic: bool = Field(
        ..., description="True when ≥2 syllable groups match (richer rhyme)."
    )


class FlowMetadata(BaseModel):
    """Diagnostic data from beat-sync analysis."""
    tempo_bpm: float
    on_beat_ratio: float = Field(..., description="Fraction of word onsets landing on a beat.")
    avg_deviation_ms: float = Field(..., description="Mean ms distance from nearest beat.")
    words_analyzed: int
    syllable_rate: Optional[float] = Field(default=None, description="Syllables per second.")
    complexity_bonus: Optional[float] = Field(default=None, description="Flow complexity bonus.")
    cadence_variance: Optional[float] = Field(default=None, description="Standard deviation of syllable rate across intervals.")
    flow_switch_bonus: Optional[float] = Field(default=None, description="Flow switch / cadence versatility bonus.")
    separation_skipped_reason: Optional[str] = Field(
        default=None,
        description=(
            "Set when Demucs vocal/accompaniment separation was skipped or "
            "failed, so flow scoring ran on mixed (non-isolated) audio "
            "instead of clean vocals. None means separation succeeded."
        ),
    )


# ── Response ──────────────────────────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    # ── Core scores ──────────────────────────────────────────
    syllable_score: float = Field(..., ge=0, le=100)
    rhyme_score: float = Field(..., ge=0, le=100)
    vocabulary_score: float = Field(..., ge=0, le=100)
    flow_score: Optional[float] = Field(
        default=None,
        description="Beat-sync score (0-100). null when audio not provided.",
    )
    total_score: float = Field(..., ge=0, le=100)

    # ── New scores ───────────────────────────────────────────
    wordplay_score: float = Field(..., ge=0, le=100)
    syllable_weight: float = Field(..., ge=0, le=100)

    # ── Semantic scores (Hindi BERT / MuRIL, via raprank-semantic) ───────────
    # null when the semantic service is unavailable -- these are an additive
    # enrichment layer and do NOT (yet) feed total_score.
    coherence_score: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Adjacent-bar semantic connectedness (0-100). null if service down.",
    )
    semantic_surprisal_score: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Word-choice unexpectedness / cleverness proxy (0-100). null if service down.",
    )
    lexical_sophistication_score: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Semantic vocabulary spread beyond surface TTR (0-100). null if service down.",
    )
    theme_consistency_score: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="How tightly lines hold the central theme (0-100). null if service down.",
    )
    callback_score: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Motif/callback reuse across distant lines (0-100). null if service down.",
    )

    # ── Literary devices (Mistral, from story/structure analysis) ────────────
    punchline_count: int = 0
    extended_metaphor_count: int = 0

    # ── Prosody / structural axes (services/prosody_service.py) ──────────────
    codeswitch_score: Optional[float] = Field(default=None, ge=0, le=100)
    repetition_score: Optional[float] = Field(default=None, ge=0, le=100)
    cadence_text_score: Optional[float] = Field(default=None, ge=0, le=100)

    # ── Style clustering (GMM, services/gmm_style_service.py) ─────────────────
    # Descriptive style fingerprint, NOT a quality score. null when the model is
    # untrained/absent. cluster = dominant style; membership = soft split.
    style_cluster: Optional[str] = Field(
        default=None, description="Dominant style cluster label (e.g. 'wordplay / entendre')."
    )
    style_cluster_confidence: Optional[float] = Field(
        default=None, ge=0, le=1, description="Posterior probability of the dominant cluster (0-1)."
    )
    style_membership: Optional[dict] = Field(
        default=None, description="Soft membership {cluster_label: probability}."
    )

    # ── Per-element clustering (services/element_cluster_service.py) ─────────
    # Descriptive, NOT a quality score. One entry per family that scored
    # successfully ("rhyme","wordplay","texture","rare"); a family is simply
    # omitted if its model was unavailable.
    element_clusters: Optional[dict] = Field(
        default=None,
        description="Per-family style fingerprints: {family: {cluster, confidence, membership}}.",
    )

    # ── Quality tier (RF, services/rf_quality_service.py) ─────────────────────
    # Supervised quality-tier classification, a comparison head alongside the
    # Bayesian/SVM scorers. null when the model is untrained/absent.
    predicted_tier: Optional[str] = Field(
        default=None, description="RF-predicted quality tier (e.g. 'commercial', 'mid', 'elite')."
    )
    tier_confidence: Optional[float] = Field(
        default=None, ge=0, le=1, description="Posterior probability of the predicted tier (0-1)."
    )
    tier_probabilities: Optional[dict] = Field(
        default=None, description="Full class distribution {tier_label: probability}."
    )

    # ── Quality tier: SVM comparison head (services/svm_quality_service.py) ──
    svm_tier: Optional[str] = Field(
        default=None, description="SVM-predicted quality tier."
    )
    svm_tier_confidence: Optional[float] = Field(
        default=None, ge=0, le=1, description="Posterior probability of the SVM tier (0-1)."
    )
    svm_tier_probabilities: Optional[dict] = Field(
        default=None, description="SVM class distribution {tier_label: probability}."
    )

    # ── Quality tier: Bayesian comparison head (services/bayesian_scoring_service.py) ──
    bayes_tier: Optional[str] = Field(
        default=None, description="Bayesian-network-predicted quality tier (argmax posterior)."
    )
    bayes_tier_probabilities: Optional[dict] = Field(
        default=None, description="Bayesian posterior {tier_label: probability}."
    )

    # ── Quality tier: consensus across the three heads ────────────────────────
    tier_consensus: Optional[str] = Field(
        default=None, description="Majority-vote tier across RF/SVM/Bayesian heads (ties -> RF)."
    )
    tier_consensus_agreement: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Fraction of available heads agreeing with the consensus (e.g. 0.67 = 2/3).",
    )

    # ── Quality tier: DPST dual-tower neural classifier ───────────────────────
    # Phonetic Transformer (Tower A) + Character CNN (Tower B) fused via
    # cross-attention. Trained end-to-end on the DHH phonetic dataset.
    # null when model weights are absent.
    dpst_tier: Optional[str] = Field(
        default=None, description="DPST-predicted quality tier (elite | mid | commercial)."
    )
    dpst_tier_confidence: Optional[float] = Field(
        default=None, ge=0, le=1, description="DPST softmax probability of the predicted tier (0-1)."
    )
    dpst_tier_probabilities: Optional[dict] = Field(
        default=None, description="DPST full class distribution {tier_label: probability}."
    )

    # ── Stats ─────────────────────────────────────────────────
    word_count: int
    line_count: int
    avg_syllables_per_word: float
    vocabulary_uniqueness: float = Field(..., description="Type-token ratio (0-1).")
    detected_language: str = Field(..., description="'en', 'hi', or 'mixed'.")

    # ── Wordplay stats ───────────────────────────────────────
    double_entendres_count: int
    puns_count: int
    similes_count: int
    metaphors_count: int
    allusions_count: int

    # ── Sound scores ─────────────────────────────────────────
    assonance_score: float = Field(..., ge=0, le=100)
    consonance_score: float = Field(..., ge=0, le=100)
    onomatopoeia_score: float = Field(..., ge=0, le=100)

    # ── Detail lists ─────────────────────────────────────────
    assonance_pairs: List[str]
    consonance_pairs: List[str]
    onomatopoeia_hits: List[str]
    rhyme_pairs: List[RhymeMatch]
    multisyllabic_rhyme_count: int
    flow_metadata: Optional[FlowMetadata] = None
    generated_lyrics: Optional[str] = Field(default=None, description="Mistral formatted lyrics.")
    story_structure: Optional[dict] = Field(
        default=None,
        description=(
            "Qualitative story/structure description from Mistral (keys: theme, "
            "story, structure, mood). Descriptive only -- Mistral no longer "
            "contributes to any score; all scoring is local and rule-based."
        ),
    )

