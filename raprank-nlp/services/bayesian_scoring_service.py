"""
Bayesian-network quality-tier scoring, learned primarily from the real,
domain-matched Indian rap corpus in corpus/data/ (the 302-song corpus from
corpus/scrape_corpus.py, tier-labeled via corpus/real_corpus/indian_tiers.py
against corpus/artists.py's expected_profile priors), with the synthetic
corpus in corpus/synthetic_data/ (see corpus/synthetic/generate.py) folded
in too when present -- its per-song generation-target labels are more
precise than the Indian corpus's artist-level ones, even though there are
far fewer of them.

An English-language real corpus (corpus/real_corpus/data/, ~7.7k songs from
35 US rap artists -- see corpus/real_corpus/fetch_real_corpus.py) is also
available but NOT included by default: this product scores Hindi/Hinglish
rap specifically, and English rhyme/syllable/vocabulary norms don't
transfer -- pass --include-us-corpus to fold it in anyway (e.g. for a
volume experiment), never as the primary source.

Every current axis score in this pipeline (rhyme, syllable, alliteration,
vocabulary, wordplay, assonance, consonance, onomatopoeia, compound_dens,
holorime_dens -- the same set corpus/analysis/signature.py already computes)
becomes an observed node; `quality_tier` (elite/mid/commercial) is the single
latent node every axis is conditioned on (naive-Bayes structure). Unlike
MAIN_WEIGHTS/RHYME["WEIGHTS"]/HYBRID_COMBINATION_WEIGHTS in
config/scoring_config.py, these conditional probability tables are *learned*
from data rather than hand-tuned, and inference produces a calibrated
posterior over tiers instead of one linear point estimate.

This module intentionally does NOT replace main.py's linear total_score --
per the plan, the two are meant to be compared side by side before deciding
whether to cut over.

Usage:
    python -m services.bayesian_scoring_service --train        # fit + save from Indian real (+ synthetic) corpus
    python -m services.bayesian_scoring_service --train --include-us-corpus  # + English corpus (off-domain, opt-in)
    python -m services.bayesian_scoring_service --eval-corpus  # posterior for every training sample
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AXES: tuple[str, ...] = (
    "rhyme", "syllable", "alliteration", "vocabulary", "wordplay",
    "assonance", "consonance", "onomatopoeia", "compound_dens", "holorime_dens",
)
TIER_NODE = "quality_tier"

BIN_EDGES = [-0.001, 25.0, 50.0, 75.0, 100.001]
BIN_LABELS = ["low", "mid", "high", "elite"]

SYNTHETIC_DATA_DIR = Path(__file__).resolve().parent.parent / "corpus" / "synthetic_data"
INDIAN_CORPUS_DATA_DIR = Path(__file__).resolve().parent.parent / "corpus" / "data"
REAL_CORPUS_DATA_DIR = Path(__file__).resolve().parent.parent / "corpus" / "real_corpus" / "data"
MODEL_PATH = SYNTHETIC_DATA_DIR / "_bayesian_model.pkl"

# Real corpus songs are far more numerous per artist than synthetic samples
# per tier (e.g. 1189 Lil Wayne songs vs. a few dozen synthetic "mid" samples)
# and their label is only artist-level, not per-song -- cap how many any one
# artist contributes so no single prolific artist's idiosyncrasies dominate
# a whole tier's learned distribution.
REAL_CORPUS_MAX_PER_ARTIST = 150


def bin_score(value: float) -> str:
    import numpy as np

    idx = int(np.digitize([value], BIN_EDGES)[0]) - 1
    idx = max(0, min(idx, len(BIN_LABELS) - 1))
    return BIN_LABELS[idx]


def _axis_scores_from_lyrics(lyrics: str) -> dict[str, float]:
    """Same axis set as corpus/analysis/signature.py, keyed to this module's AXES names."""
    from corpus.analysis.signature import signature

    sig = signature(lyrics)
    return {
        "rhyme": sig["rhyme"],
        "syllable": sig["syl_density"],
        "alliteration": sig["alliteration"],
        "vocabulary": sig["vocab"],
        "wordplay": sig["wordplay"],
        "assonance": sig["assonance"],
        "consonance": sig["consonance"],
        "onomatopoeia": sig["onomatopoeia"],
        "compound_dens": sig["compound_dens"],
        "holorime_dens": sig["holorime_dens"],
    }


def load_synthetic_records() -> list[dict]:
    """Every accepted sample written by corpus/synthetic/generate.py."""
    records: list[dict] = []
    if not SYNTHETIC_DATA_DIR.exists():
        return records
    for path in SYNTHETIC_DATA_DIR.glob("*/*/*.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("lyrics") and rec.get("tier"):
            records.append(rec)
    return records


def load_real_corpus_records(max_per_artist: int = REAL_CORPUS_MAX_PER_ARTIST) -> list[dict]:
    """Every song written by corpus/real_corpus/fetch_real_corpus.py (the
    ~7.7k-song ENGLISH corpus), capped per artist so a single prolific artist
    can't dominate their tier. Off-domain for this product (Hindi/Hinglish
    rap) -- see load_indian_corpus_records for the domain-matched corpus.
    Callers must opt in explicitly; this is never included by load_all_records
    by default (see its include_us_corpus parameter)."""
    records: list[dict] = []
    if not REAL_CORPUS_DATA_DIR.exists():
        return records
    for artist_dir in sorted(REAL_CORPUS_DATA_DIR.iterdir()):
        if not artist_dir.is_dir():
            continue
        n = 0
        for path in sorted(artist_dir.glob("*.json")):
            if n >= max_per_artist:
                break
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rec.get("lyrics") and rec.get("tier"):
                records.append(rec)
                n += 1
    return records


def load_indian_corpus_records() -> list[dict]:
    """Every song in corpus/data/ (the 302-song scraped Hindi/Hinglish
    corpus) with a tier assigned via corpus/real_corpus/indian_tiers.py's
    artist_tier_map() -- domain-matched, unlike the English real corpus.
    This is the corpus's own real songs; it carries the same artist-level
    (not per-song) label limitation as load_real_corpus_records, but in the
    right language/rhyme-system/culture."""
    from corpus.real_corpus.indian_tiers import artist_tier_map

    records: list[dict] = []
    if not INDIAN_CORPUS_DATA_DIR.exists():
        return records
    tier_map = artist_tier_map()
    for artist_dir in sorted(INDIAN_CORPUS_DATA_DIR.iterdir()):
        if not artist_dir.is_dir():
            continue
        for path in sorted(artist_dir.glob("*.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            tier = tier_map.get(rec.get("artist"))
            if rec.get("lyrics") and tier:
                rec = {**rec, "tier": tier}
                records.append(rec)
    return records


def load_all_records(include_us_corpus: bool = False) -> list[dict]:
    """Domain-matched Indian corpus + synthetic data by default. The English
    corpus is off-domain (different language/rhyme-system/culture) and is
    only included when include_us_corpus=True is passed explicitly."""
    records = load_indian_corpus_records() + load_synthetic_records()
    if include_us_corpus:
        records += load_real_corpus_records()
    return records


def build_training_frame(records: list[dict]):
    import pandas as pd

    rows = []
    for rec in records:
        scores = _axis_scores_from_lyrics(rec["lyrics"])
        row = {axis: bin_score(scores[axis]) for axis in AXES}
        row[TIER_NODE] = rec["tier"]
        rows.append(row)
    return pd.DataFrame(rows, columns=[*AXES, TIER_NODE])


def train(records: list[dict] | None = None):
    """Fit the naive-Bayes-structured network: quality_tier -> each axis node."""
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.models import DiscreteBayesianNetwork

    if records is None:
        records = load_all_records()
    if not records:
        raise RuntimeError(
            "No training data found -- run corpus.real_corpus.fetch_real_corpus "
            "and/or corpus.synthetic.generate (needs MISTRAL_API_KEY) first."
        )

    df = build_training_frame(records)
    edges = [(TIER_NODE, axis) for axis in AXES]
    model = DiscreteBayesianNetwork(edges)
    # BDeu (Bayesian/Dirichlet) prior smooths unseen bin/tier combinations --
    # a pilot batch of a few hundred samples won't cover every bin per tier,
    # and MaximumLikelihoodEstimator would assign those zero probability.
    estimator = BayesianEstimator(model, df)
    cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=5)
    model.add_cpds(*cpds)
    return model


def save(model, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load(path: Path = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_posterior(model, scores: dict[str, float]) -> dict[str, float]:
    """Calibrated P(quality_tier | observed axis scores), given a scores dict
    keyed like AXES (e.g. from _axis_scores_from_lyrics or main.py's breakdown)."""
    from pgmpy.inference import VariableElimination

    evidence = {axis: bin_score(scores[axis]) for axis in AXES if axis in scores}
    infer = VariableElimination(model)
    result = infer.query(variables=[TIER_NODE], evidence=evidence, show_progress=False)
    return dict(zip(result.state_names[TIER_NODE], result.values.tolist()))


def score_lyrics(model, lyrics: str) -> dict[str, float]:
    return predict_posterior(model, _axis_scores_from_lyrics(lyrics))


def main() -> int:
    ap = argparse.ArgumentParser(description="Train / evaluate the Bayesian quality-tier network")
    ap.add_argument("--train", action="store_true", help="fit from Indian (+ synthetic) corpus and save")
    ap.add_argument("--eval-corpus", action="store_true", help="print posterior for every training sample")
    ap.add_argument("--include-us-corpus", action="store_true",
                     help="also fold in the off-domain English real corpus (opt-in, not recommended by default)")
    args = ap.parse_args()

    if args.train:
        records = load_all_records(include_us_corpus=args.include_us_corpus)
        indian_n = len(load_indian_corpus_records())
        synth_n = len(load_synthetic_records())
        us_n = len(records) - indian_n - synth_n
        print(f"Training on {len(records)} samples ({indian_n} Indian real + {synth_n} synthetic"
              f"{f' + {us_n} US real (opt-in)' if us_n else ''})...")
        model = train(records)
        save(model)
        print(f"Saved model to {MODEL_PATH}")
        return 0

    if args.eval_corpus:
        model = load()
        records = load_all_records(include_us_corpus=args.include_us_corpus)
        correct = 0
        for rec in records:
            posterior = score_lyrics(model, rec["lyrics"])
            predicted = max(posterior, key=posterior.get)
            correct += predicted == rec["tier"]
            print(f"  {rec['tier']:10} -> predicted={predicted:10} {posterior}")
        if records:
            print(f"\nAccuracy vs. known/assumed tier label: {correct}/{len(records)} = {correct/len(records):.0%}")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
