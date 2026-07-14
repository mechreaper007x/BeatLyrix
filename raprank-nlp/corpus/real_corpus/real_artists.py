"""
Artist -> quality-tier map for the real, third-party English rap corpus
(Cropinky/rap_lyrics_english on Hugging Face) used as a second training
source for services/bayesian_scoring_service.py, alongside the synthetic
corpus in corpus/synthetic_data/.

There is no per-song ground truth for real lyrics, so -- same limitation as
corpus/artists.py's `expected_profile` priors -- every song by an artist
inherits that artist's tier from established hip-hop critical consensus
(XXL/Rolling Stone/Complex "greatest lyricists" rankings): dense multi-
syllabic/internal rhyme technicians vs. hook-driven, plainer commercial
writers. This is a coarser label than the synthetic corpus's per-song
generation target, so treat real-corpus samples as a lower-confidence
training signal (see REAL_CORPUS_SAMPLE_WEIGHT in bayesian_scoring_service.py).

Lyric text itself is fetched at train-time from the HF dataset and cached
under corpus/real_corpus/data/ (git-ignored, local-only -- same handling as
corpus/data/, never bundled or served to end users).
"""
from __future__ import annotations

# repo folder name (as stored in the HF dataset under songs/<folder>/<Artist>.txt)
# -> (artist display name, tier)
REAL_ARTIST_TIERS: dict[str, tuple[str, str]] = {
    # elite: dense multisyllabic/internal rhyme technicians, consistently top
    # "greatest lyricist" rankings
    "flowmasteri/Nas": ("Nas", "elite"),
    "wutang/GZA": ("GZA", "elite"),
    "flowmasteri/Big L": ("Big L", "elite"),
    "flowmasteri/Big Pun": ("Big Pun", "elite"),
    "wutang/Raekwon": ("Raekwon", "elite"),
    "wutang/Ghostface Killah": ("Ghostface Killah", "elite"),
    "wutang/Inspectah Deck": ("Inspectah Deck", "elite"),
    "wutang/RZA": ("RZA", "elite"),
    "zenske/Lauryn Hill": ("Lauryn Hill", "elite"),
    "zenske/Jean Grae": ("Jean Grae", "elite"),
    "zenske/MC Lyte": ("MC Lyte", "elite"),
    "zenske/Rah Digga": ("Rah Digga", "elite"),
    "flowmasteri/The Notorious B.I.G.": ("The Notorious B.I.G.", "elite"),

    # mid: structured, competent, but not defined by dense rhyme technicality
    "flowmasteri/50 Cent": ("50 Cent", "mid"),
    "wutang/Method Man": ("Method Man", "mid"),
    "wutang/Cappadonna": ("Cappadonna", "mid"),
    "wutang/Masta Killa": ("Masta Killa", "mid"),
    "wutang/U-God": ("U-God", "mid"),
    "wutang/Wu-Tang Clan": ("Wu-Tang Clan", "mid"),
    "ye/Kanye West": ("Kanye West", "mid"),
    "zenske/Missy Elliott": ("Missy Elliott", "mid"),
    "zenske/Remy Ma": ("Remy Ma", "mid"),
    "flowmasteri/Action Bronson": ("Action Bronson", "mid"),
    "flowmasteri/Prodigy of Mobb Deep": ("Prodigy of Mobb Deep", "mid"),
    "flowmasteri/ILL BILL": ("ILL BILL", "mid"),
    "wutang/Ol’ Dirty Bastard": ("Ol' Dirty Bastard", "mid"),
    "zenske/Lil’ Kim": ("Lil' Kim", "mid"),
    "Lilgpt/Lil Wayne": ("Lil Wayne", "mid"),

    # commercial: hook-driven, simple end-rhyme, low device density (trap/mumble era)
    "Lilgpt/Lil Baby": ("Lil Baby", "commercial"),
    "Lilgpt/Lil Durk": ("Lil Durk", "commercial"),
    "Lilgpt/Lil Flip": ("Lil Flip", "commercial"),
    "Lilgpt/Lil Peep": ("Lil Peep", "commercial"),
    "Lilgpt/Lil Pump": ("Lil Pump", "commercial"),
    "Lilgpt/Lil Reese": ("Lil Reese", "commercial"),
    "Lilgpt/Lil Uzi Vert": ("Lil Uzi Vert", "commercial"),
}


def tier_for(repo_path: str) -> tuple[str, str] | None:
    return REAL_ARTIST_TIERS.get(repo_path)
