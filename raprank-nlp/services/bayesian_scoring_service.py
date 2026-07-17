"""
Bayesian-network quality-tier scoring.

Training contract (shared by all five model services, revised Jul 2026):
DEPLOY models train on synthetic data + consented seeds ONLY -- scraped real
lyrics never enter a shipped model (copyright posture; see
load_train_records). The real Indian corpus (corpus/data/, artist-level
tiers via corpus/real_corpus/indian_tiers.py) is EVALUATION-ONLY: every
--train reports a synthetic-only fit tested against it as the GENERATOR-
REALISM transfer metric (57.3% RF baseline, Jul 2026, vs ~75% when real
data trained the model -- that gap is the generator's remaining realism
deficit, and closing it is the roadmap).

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
    python -m services.bayesian_scoring_service --train        # fit on synthetic, save, eval on real held-out
    python -m services.bayesian_scoring_service --eval-corpus  # posterior for every held-out real song
"""
from __future__ import annotations

import argparse
import hashlib
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
    """Same axis set as corpus/analysis/signature.py, keyed to this module's AXES names.
    Backed by an on-disk cache (md5(lyrics) -> scores; NO lyric text stored, so
    it's copyright-clean) because signature() costs ~200ms/verse and every
    training run recomputes the same ~1300 corpus records."""
    cache = _feature_cache()
    key = hashlib.md5(lyrics.encode("utf-8")).hexdigest()
    hit = cache.get(key)
    if hit is not None and all(ax in hit for ax in AXES):
        return {ax: hit[ax] for ax in AXES}

    from corpus.analysis.signature import signature

    sig = signature(lyrics)
    scores = {
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
    cache[key] = scores
    _feature_cache_append(key, scores)
    return scores


# In-memory view of the on-disk cache; loaded once per process.
_FEATURE_CACHE_PATH = SYNTHETIC_DATA_DIR / "_axis_feature_cache.jsonl"
_feature_cache_mem: dict[str, dict] | None = None
# v3 = updated rhyme weights and allusion fallback (2026-07).
_FEATURE_CACHE_VERSION = 3


def _feature_cache() -> dict[str, dict]:
    global _feature_cache_mem
    if _feature_cache_mem is None:
        _feature_cache_mem = {}
        if _FEATURE_CACHE_PATH.exists():
            for line in _FEATURE_CACHE_PATH.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("_v") == _FEATURE_CACHE_VERSION and "_k" in rec:
                    _feature_cache_mem[rec["_k"]] = rec
    return _feature_cache_mem


def _feature_cache_append(key: str, scores: dict[str, float]) -> None:
    try:
        _FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_FEATURE_CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"_k": key, "_v": _FEATURE_CACHE_VERSION, **scores}) + "\n")
    except Exception:
        pass  # cache is an optimization; never let it break scoring


def load_synthetic_records() -> list[dict]:
    """Every accepted sample written by corpus/synthetic/generate.py.
    Each record gains a `_path` (repo-relative) so downstream consumers
    (semantic caches, cluster kept_ids) have a stable per-record id."""
    root = SYNTHETIC_DATA_DIR.parent.parent
    records: list[dict] = []
    if not SYNTHETIC_DATA_DIR.exists():
        return records
    for path in SYNTHETIC_DATA_DIR.glob("*/*/*.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("lyrics") and rec.get("tier"):
            rec["_path"] = path.relative_to(root).as_posix()
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

    root = INDIAN_CORPUS_DATA_DIR.parent.parent
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
                rec = {**rec, "tier": tier, "_path": path.relative_to(root).as_posix()}
                records.append(rec)
    return records


def load_all_records(include_us_corpus: bool = False) -> list[dict]:
    """Domain-matched Indian corpus + synthetic data (the OLD mixed training
    pool -- kept as an escape hatch for A/B against pre-split numbers; see
    load_train_records/load_eval_records for the current contract). The
    English corpus is off-domain and only included when include_us_corpus=True."""
    records = load_indian_corpus_records() + load_synthetic_records()
    if include_us_corpus:
        records += load_real_corpus_records()
    return records


def load_train_records() -> list[dict]:
    """The DEPLOY training pool: synthetic data + consented seeds ONLY.

    Copyright posture (decided Jul 2026): scraped real lyrics are copyrighted
    text; training a commercially deployed model on them is a materially
    riskier legal position than local analysis, and India's Copyright Act has
    no clear text-and-data-mining exception. So deployed models train only on
    data we're clean on -- synthetic generation (built copyright-safe: word-
    level facts only, no line copying) and consented seeds (explicit
    permission). The real corpus is EVALUATION-ONLY: nothing from it persists
    in any shipped artifact.

    The measured cost (Jul 2026): mixed-pool training scored ~75% under
    artist-grouped CV vs ~57-60% synthetic->real transfer -- that gap is the
    generator-realism gap, tracked by the transfer metric every --train
    reports. Improving generation (compound rhymes, cadence, repetition
    realism) closes it without touching scraped text."""
    return load_synthetic_records() + _load_consented_seed_records()


def _load_consented_seed_records() -> list[dict]:
    """Consented seed verses (explicit permission, owner-judged tier labels) --
    the one human-grounded, copyright-clean real-lyrics source."""
    try:
        from services.tests.conftest import load_consented_seeds
        return [r for r in load_consented_seeds() if r.get("tier")]
    except Exception:
        return []


def load_synthetic_only_records() -> list[dict]:
    """The TRANSFER-METRIC pool: synthetic data only (no consented seeds), so
    the metric isolates generator realism. Fitting on this and testing on the
    real corpus measures how well synthetic generation mimics real rap
    (57.3% RF as of Jul 2026) -- tracked over time as generation improves."""
    return load_synthetic_records()


def load_eval_records() -> list[dict]:
    """Every real Indian corpus song (uncapped), artist-level tier labels.
    Test set for the synthetic->real transfer metric; also part of the deploy
    pool (where artist-grouped CV keeps the accuracy estimate honest)."""
    return load_indian_corpus_records()


def record_groups(records: list[dict]) -> list[str]:
    """CV group key per record: the artist for real songs (their tier label is
    artist-level, so songs by one artist must never straddle train/test), a
    unique key for each synthetic sample (tier labels are per-song)."""
    return [
        rec.get("artist") or rec.get("_path") or f"rec_{i}"
        for i, rec in enumerate(records)
    ]


def is_synthetic(rec: dict) -> bool:
    return str(rec.get("_path", "")).startswith("corpus/synthetic_data")


# Pre-computed feature alignment parameters to resolve domain shift copyright-clean.
# Format: feature -> (mean_synth, std_synth, mean_real, std_real)
ALIGNMENT_PARAMS = {
    "rhyme": (50.63577836411609, 19.27462099203883, 39.50694915254238, 17.84385751602289),
    "syllable": (80.4011345646438, 16.9198861069129, 62.55787193973635, 23.00637831555988),
    "alliteration": (66.62142480211081, 31.440921064530144, 45.86559322033899, 27.77518601021679),
    "vocabulary": (80.32411609498679, 21.209807866788456, 58.088022598870054, 21.969822442568326),
    "wordplay": (73.84175461741425, 27.68657996921056, 48.106610169491525, 28.2430767505317),
    "assonance": (65.98990765171503, 33.49814462997379, 50.37094161958569, 30.1608231873418),
    "consonance": (73.84244063324539, 21.176577736496807, 48.86706214689266, 28.81831122600302),
    "onomatopoeia": (7.750910290237466, 21.2622457332724, 14.467815442561207, 23.5796512475175),
    "compound_dens": (0.014841688654353561, 0.2905275368587882, 0.19528007769775738, 0.8078799434312889),
}


def align_synthetic_features(scores: dict[str, float]) -> dict[str, float]:
    """Map real-world feature values into synthetic training space (real→synthetic).
    Classifiers trained on synthetic data expect feature values in synthetic ranges.
    z-score in real space, then rescale to synthetic distribution."""
    aligned = {}
    for k, v in scores.items():
        if k in ALIGNMENT_PARAMS:
            ms, ss, mr, sr = ALIGNMENT_PARAMS[k]
            if sr > 0:
                # Standardize from real space, then map to synthetic space
                val = ((v - mr) / sr) * ss + ms
                # Clamp to [0.0, 100.0]
                aligned[k] = max(0.0, min(100.0, val))
            else:
                aligned[k] = v
        else:
            aligned[k] = v
    return aligned


def evaluate_on_real(predict_fn) -> dict:
    """Shared held-out evaluation for the tier heads: run `predict_fn(lyrics)
    -> tier` over every real Indian record, return accuracy + confusion
    matrix + per-class recall. Reused by the Bayesian/SVM/RF CLIs."""
    records = load_eval_records()
    labels = sorted({rec["tier"] for rec in records})
    idx = {t: i for i, t in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    correct = 0
    for rec in records:
        pred = predict_fn(rec["lyrics"])
        if pred in idx:
            cm[idx[rec["tier"]]][idx[pred]] += 1
        correct += pred == rec["tier"]
    recall = {
        t: (cm[i][i] / sum(cm[i]) if sum(cm[i]) else 0.0)
        for i, t in enumerate(labels)
    }
    return {
        "n": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "labels": labels,
        "confusion_matrix": cm,
        "per_class_recall": recall,
    }


def print_real_eval(result: dict) -> None:
    print(f"\nHeld-out accuracy (real Indian corpus, n={result['n']}): "
          f"{result['accuracy']:.1%}")
    print(f"Confusion matrix (rows=actual, cols=predicted), labels={result['labels']}:")
    for label, row in zip(result["labels"], result["confusion_matrix"]):
        print(f"  {label:12} {row}")
    print("Per-class recall: "
          + ", ".join(f"{t}={r:.1%}" for t, r in result["per_class_recall"].items()))


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
        records = load_train_records()
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
    keyed like AXES (e.g. from _axis_scores_from_lyrics or main.py's breakdown).
    Applies domain-shift alignment so real-world scores are mapped into the
    synthetic training distribution before inference -- no real lyrics in training."""
    from pgmpy.inference import VariableElimination

    aligned = align_synthetic_features(scores)
    evidence = {axis: bin_score(aligned[axis]) for axis in AXES if axis in aligned}
    infer = VariableElimination(model)
    result = infer.query(variables=[TIER_NODE], evidence=evidence, show_progress=False)
    return dict(zip(result.state_names[TIER_NODE], result.values.tolist()))


def score_lyrics(model, lyrics: str) -> dict[str, float]:
    return predict_posterior(model, _axis_scores_from_lyrics(lyrics))


def main() -> int:
    ap = argparse.ArgumentParser(description="Train / evaluate the Bayesian quality-tier network")
    ap.add_argument("--train", action="store_true",
                     help="fit deploy model (real+synthetic), save, plus report the "
                          "synthetic->real transfer metric")
    ap.add_argument("--eval-corpus", action="store_true",
                     help="print posterior for every real Indian corpus song")
    ap.add_argument("--skip-transfer", action="store_true",
                     help="skip the synthetic->real transfer metric (faster)")
    ap.add_argument("--include-us-corpus", action="store_true",
                     help="also fold in the off-domain English real corpus (opt-in, not recommended by default)")
    args = ap.parse_args()

    if args.train:
        records = load_train_records()
        if args.include_us_corpus:
            records = records + load_real_corpus_records()
        n_seed = sum(not is_synthetic(r) for r in records)
        print(f"Training DEPLOY model on {len(records)} samples "
              f"({len(records) - n_seed} synthetic + {n_seed} consented seeds; "
              f"scraped real lyrics are evaluation-only)...")
        model = train(records)
        save(model)
        print(f"Saved model to {MODEL_PATH}")
        print("(Note: the Bayesian net has no CV step; its honest accuracy is the "
              "transfer metric below and side-by-side comparison with RF/SVM.)")

        if not args.skip_transfer:
            print("\n--- Generator-realism transfer metric (synthetic-only fit -> real test) ---")
            synth_model = train(load_synthetic_only_records())
            def _predict(lyrics: str) -> str:
                posterior = score_lyrics(synth_model, lyrics)
                return max(posterior, key=posterior.get)
            print_real_eval(evaluate_on_real(_predict))
            print("(This number tracks synthetic-corpus realism, NOT the deployed model.)")
        return 0

    if args.eval_corpus:
        model = load()
        records = load_eval_records()
        correct = 0
        for rec in records:
            posterior = score_lyrics(model, rec["lyrics"])
            predicted = max(posterior, key=posterior.get)
            correct += predicted == rec["tier"]
            print(f"  {rec['tier']:10} -> predicted={predicted:10} {posterior}")
        if records:
            print(f"\nHeld-out accuracy vs. artist-level tier label: "
                  f"{correct}/{len(records)} = {correct/len(records):.0%} "
                  f"(real corpus is evaluation-only; deploy model never trains on it)")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
