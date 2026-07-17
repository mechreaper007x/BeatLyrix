"""
GMM style-clustering: unsupervised discovery of rap STYLE modes (what KIND of
rapper), distinct from the percentile calibration that scores HOW GOOD a verse
is. Each corpus song becomes a 26-dim feature vector -- the 22 local technical
axes from corpus/analysis/signature.py PLUS the 4 semantic raw metrics from the
raprank-semantic Space -- and a Gaussian Mixture (k chosen by BIC) partitions
them into soft style clusters. A new verse gets a soft membership like
"72% dense-lyrical, 28% commercial-melodic".

GMM (not percentiles) is correct HERE because this is genuine latent-mode
clustering, not a monotonic score. total_score is NOT affected -- clustering is
descriptive.

Training contract (shared by all five model services): TRAIN on the synthetic
corpus only (semantic metrics live-fetched from the Space on first run and
persisted to _synthetic_semantic_cache.jsonl); the real Indian corpus is the
held-out evaluation pool (avg log-likelihood / BIC / assignment table).

Mirrors the train/save/load/predict + argparse shape of
services/bayesian_scoring_service.py. scikit-learn is imported lazily so a
missing install never breaks the /analyze request path.

Usage:
    python -m services.gmm_style_service --train        # fit on synthetic (live-fetch), eval on real
    python -m services.gmm_style_service --report       # human-readable cluster labels
    python -m services.gmm_style_service --eval-corpus   # per-song assignment
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Feature layout ───────────────────────────────────────────────────────────
# Technical axes come straight from signature() (same names/order as
# corpus/analysis/signature.py AXES). Semantic axes are the 4 raw metrics the
# semantic service returns.
TECHNICAL_AXES: tuple[str, ...] = (
    "syl_density", "syl_weight", "rhyme", "internal", "chain", "multi_dens",
    "compound_dens", "holorime_dens", "alliteration", "assonance", "consonance",
    "onomatopoeia", "vocab", "wordplay", "simile", "metaphor", "pun", "entendre",
    "english", "codeswitch", "repetition", "cadence_var",
)
SEMANTIC_AXES: tuple[str, ...] = (
    "coherence_cosine", "theme_cosine", "pairwise_spread", "mean_surprisal_nats",
)
FEATURE_NAMES: tuple[str, ...] = TECHNICAL_AXES + SEMANTIC_AXES

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "corpus" / "synthetic_data" / "_gmm_style_model.pkl"
REPORT_PATH = ROOT / "corpus" / "synthetic_data" / "_gmm_style_report.json"
# Semantic raw metrics collected once by raprank-semantic/build_calibration.py.
SEMANTIC_CACHE = ROOT.parent / "raprank-semantic" / "calibration_raw.jsonl"
# Live-fetched semantic metrics for SYNTHETIC tracks (not covered by the
# calibration cache) are persisted here so retraining never refetches -- one
# jsonl line per track, keyed by the record's repo-relative _path.
SYNTHETIC_SEMANTIC_CACHE = ROOT / "corpus" / "synthetic_data" / "_synthetic_semantic_cache.jsonl"

# Candidate cluster counts. Capped at 6: on 302 songs, BIC will happily split
# off 2-song micro-clusters around single-feature outliers (e.g. a compound_dens
# spike) if allowed to go higher -- those aren't real styles. 2-6 keeps clusters
# macro and interpretable.
BIC_K_RANGE = range(2, 7)

# Human-readable descriptor per axis, used by label_clusters() to build cluster
# names like "Wordplay-Dense · Rhyme-Heavy" instead of raw axis names like
# "wordplay / rhyme". Covers every TECHNICAL_AXES + SEMANTIC_AXES name.
AXIS_DESCRIPTORS: dict[str, str] = {
    "syl_density": "Syllable-Dense",
    "syl_weight": "Multisyllabic",
    "rhyme": "Rhyme-Heavy",
    "internal": "Internal-Rhyme-Rich",
    "chain": "Rhyme-Chain-Heavy",
    "multi_dens": "Multi-Rhyme-Dense",
    "compound_dens": "Compound-Rhyme-Dense",
    "holorime_dens": "Holorime-Dense",
    "alliteration": "Alliterative",
    "assonance": "Assonant",
    "consonance": "Consonant-Rich",
    "onomatopoeia": "Onomatopoeic",
    "vocab": "Vocabulary-Rich",
    "wordplay": "Wordplay-Dense",
    "simile": "Simile-Heavy",
    "metaphor": "Metaphor-Heavy",
    "pun": "Pun-Heavy",
    "entendre": "Double-Entendre-Heavy",
    "english": "English-Leaning",
    "codeswitch": "Code-Switching",
    "repetition": "Repetition-Driven",
    "cadence_var": "Cadence-Varied",
    "coherence_cosine": "Coherent",
    "theme_cosine": "Thematic",
    "pairwise_spread": "Lexically-Sophisticated",
    "mean_surprisal_nats": "Unpredictable",
}


# ── Feature extraction ───────────────────────────────────────────────────────
def _technical_features(lyrics: str) -> dict[str, float]:
    from corpus.analysis.signature import signature
    sig = signature(lyrics)
    return {ax: float(sig.get(ax, 0.0)) for ax in TECHNICAL_AXES}


def _load_semantic_cache() -> dict[str, dict]:
    """id -> {metric: value}, merged from build_calibration's log (real corpus,
    'artist-slug/title-slug' ids) and the synthetic live-fetch cache (repo-
    relative _path ids)."""
    out: dict[str, dict] = {}
    for cache_path in (SEMANTIC_CACHE, SYNTHETIC_SEMANTIC_CACHE):
        if not cache_path.exists():
            continue
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "_id" in rec:
                out[rec["_id"]] = rec
    return out


def _cache_id_from_path(path_str: str) -> str:
    """corpus/data/<artist>/<title>.json -> '<artist>/<title>' (matches the cache _id)."""
    p = Path(path_str)
    return f"{p.parent.name}/{p.stem}"


def _semantic_features(song: dict, cache: dict[str, dict], live_fetch: bool) -> dict[str, float] | None:
    # Synthetic records cache under their full repo-relative _path; real
    # corpus songs under the calibration log's 'artist/title' id.
    path = song.get("_path", "")
    rec = cache.get(path) or cache.get(_cache_id_from_path(path))
    if rec is None and live_fetch:
        rec = _fetch_semantic_live(song["lyrics"])
        if rec is not None and path:
            try:  # persist so retraining never refetches this track
                with open(SYNTHETIC_SEMANTIC_CACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"_id": path, **rec}, ensure_ascii=False) + "\n")
            except Exception:
                pass
    if rec is None:
        return None
    try:
        return {ax: float(rec[ax]) for ax in SEMANTIC_AXES}
    except (KeyError, TypeError, ValueError):
        return None


def _fetch_semantic_live(lyrics: str) -> dict | None:
    """Fallback: pull raw metrics from the live Space for a song not in the cache."""
    import os
    import httpx
    url = os.getenv("SEMANTIC_API_URL", "https://mechreaper007x-raprank-semantic.hf.space").rstrip("/")
    try:
        r = httpx.post(f"{url}/semantic", json={"lyrics": lyrics}, timeout=120.0)
        r.raise_for_status()
        return r.json().get("metrics", {})
    except Exception:
        return None


def build_feature_matrix(records: list[dict], live_fetch: bool = False):
    """Return (X ndarray, kept_records) -- songs missing semantic data are dropped."""
    import numpy as np
    cache = _load_semantic_cache()
    rows, kept = [], []
    dropped = 0
    for i, song in enumerate(records):
        if live_fetch and i and i % 25 == 0:
            print(f"  ...{i}/{len(records)} processed ({len(kept)} kept, {dropped} dropped)")
        sem = _semantic_features(song, cache, live_fetch)
        if sem is None:
            dropped += 1
            continue
        tech = _technical_features(song["lyrics"])
        rows.append([tech[a] for a in TECHNICAL_AXES] + [sem[a] for a in SEMANTIC_AXES])
        kept.append(song)
    if dropped:
        print(f"  build_feature_matrix: kept {len(kept)}, dropped {dropped} (no semantic metrics)")
    return np.asarray(rows, dtype=float), kept


# ── Train / persist ──────────────────────────────────────────────────────────
def train(records: list[dict] | None = None, live_fetch: bool = False) -> dict:
    """Standardize features, pick k by BIC, fit a diagonal-covariance GMM.
    Returns a bundle dict {scaler, gmm, feature_names, k, assignments, kept_ids}."""
    import numpy as np
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    if records is None:
        records = _load_train_records()
    X, kept = build_feature_matrix(records, live_fetch=live_fetch)
    if len(X) < max(BIC_K_RANGE) * 3:
        raise RuntimeError(
            f"Only {len(X)} songs had both technical + semantic features -- "
            f"run with live_fetch=True (synthetic tracks) or "
            f"raprank-semantic/build_calibration.py (real corpus) first."
        )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best = None
    for k in BIC_K_RANGE:
        gm = GaussianMixture(
            n_components=k, covariance_type="diag",
            random_state=0, n_init=5, max_iter=300,
        ).fit(Xs)
        bic = gm.bic(Xs)
        if best is None or bic < best[0]:
            best = (bic, k, gm)
    _, k, gmm = best

    assignments = gmm.predict(Xs).tolist()
    return {
        "scaler": scaler,
        "gmm": gmm,
        "feature_names": list(FEATURE_NAMES),
        "k": k,
        "assignments": assignments,
        "kept_ids": [_record_id(s) for s in kept],
        "kept_artists": [s.get("artist", s.get("tier", "?")) for s in kept],
    }


# ── K-means comparison ───────────────────────────────────────────────────────
# GMM was picked mainly to mirror the existing Bayesian-net/style-of-code in
# this project, not because it's provably the best clusterer for 26 dims on
# ~300 songs. Silhouette score (cluster separation/cohesion, independent of
# BIC's likelihood-based criterion) lets us check that empirically: fit both
# GMM (hard-argmax labels) and K-means over the same standardized features
# and the same k range, and compare. If K-means silhouette is meaningfully
# higher, that's a signal GMM's soft/elliptical assumption isn't earning its
# extra complexity here.
def compare_kmeans(records: list[dict] | None = None, live_fetch: bool = False) -> dict:
    """Fit GMM and K-means on identical standardized features, report
    silhouette score for each at their own BIC/inertia-elbow-picked k, plus
    K-means at the GMM's chosen k for a same-k apples-to-apples comparison."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    if records is None:
        records = _load_corpus_records()
    X, kept = build_feature_matrix(records, live_fetch=live_fetch)
    if len(X) < max(BIC_K_RANGE) * 3:
        raise RuntimeError(
            f"Only {len(X)} songs had both technical + semantic features -- "
            f"run raprank-semantic/build_calibration.py first (needs the cache)."
        )
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    gmm_best = None
    for k in BIC_K_RANGE:
        gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=0, n_init=5, max_iter=300).fit(Xs)
        bic = gm.bic(Xs)
        if gmm_best is None or bic < gmm_best[0]:
            gmm_best = (bic, k, gm)
    _, gmm_k, gmm = gmm_best
    gmm_labels = gmm.predict(Xs)
    gmm_silhouette = float(silhouette_score(Xs, gmm_labels))

    # K-means picks k by its own criterion (best silhouette over the same
    # range) rather than borrowing GMM's BIC-picked k, so it isn't handicapped
    # by a criterion built for a different model.
    km_best = None
    for k in BIC_K_RANGE:
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
        sil = silhouette_score(Xs, km.labels_)
        if km_best is None or sil > km_best[0]:
            km_best = (sil, k, km)
    km_silhouette, km_k, km = km_best

    # Same-k comparison: how does K-means do at exactly GMM's chosen k.
    km_at_gmm_k = KMeans(n_clusters=gmm_k, random_state=0, n_init=10).fit(Xs)
    km_at_gmm_k_silhouette = float(silhouette_score(Xs, km_at_gmm_k.labels_))

    return {
        "n_songs": len(X),
        "gmm": {"k": gmm_k, "silhouette": gmm_silhouette},
        "kmeans_own_k": {"k": km_k, "silhouette": float(km_silhouette)},
        "kmeans_at_gmm_k": {"k": gmm_k, "silhouette": km_at_gmm_k_silhouette},
    }


def save(bundle: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load(path: Path = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Cluster labelling (human-readable) ───────────────────────────────────────
def label_clusters(bundle: dict) -> dict:
    """Per cluster: top distinguishing axes (largest standardized centroid
    components) + dominant artists. Produces a short human label."""
    import numpy as np
    from collections import Counter

    gmm, names = bundle["gmm"], bundle["feature_names"]
    means = gmm.means_                      # (k, 26), already in standardized space
    assignments = bundle["assignments"]
    artists = bundle["kept_artists"]

    clusters = {}
    for c in range(bundle["k"]):
        centroid = means[c]
        top_idx = np.argsort(centroid)[::-1][:4]           # 4 highest axes
        top_axes = [(names[i], round(float(centroid[i]), 2)) for i in top_idx]
        members = [artists[i] for i, a in enumerate(assignments) if a == c]
        dom_artists = Counter(members).most_common(4)
        descriptors = [AXIS_DESCRIPTORS.get(ax, ax) for ax, _ in top_axes[:2]]
        label = " · ".join(descriptors)
        clusters[str(c)] = {
            "label": label,
            "size": len(members),
            "top_axes": top_axes,
            "dominant_artists": dom_artists,
        }
    return clusters


# ── Inference (runtime) ──────────────────────────────────────────────────────
def score_style(lyrics: str, semantic_metrics: dict | None = None, bundle: dict | None = None) -> dict | None:
    """Soft style membership for a verse. semantic_metrics: the raw `metrics`
    dict from analyze_semantics (reused so no extra Space call); if None we try a
    live fetch. Returns None on any failure (missing model / sklearn / semantics)."""
    try:
        import numpy as np
        if bundle is None:
            bundle = load()
        if semantic_metrics is None:
            semantic_metrics = _fetch_semantic_live(lyrics)
        if not semantic_metrics:
            return None
        try:
            sem = [float(semantic_metrics[a]) for a in SEMANTIC_AXES]
        except (KeyError, TypeError, ValueError):
            return None

        tech = _technical_features(lyrics)
        vec = np.asarray([[tech[a] for a in TECHNICAL_AXES] + sem], dtype=float)
        Xs = bundle["scaler"].transform(vec)
        probs = bundle["gmm"].predict_proba(Xs)[0]

        labels = bundle.get("_labels")  # attached by --report; may be absent
        membership = {}
        for c, p in enumerate(probs):
            name = labels[str(c)]["label"] if labels else f"cluster_{c}"
            membership[name] = round(float(p), 4)
        top = max(range(len(probs)), key=lambda i: probs[i])
        top_name = labels[str(top)]["label"] if labels else f"cluster_{top}"
        return {
            "cluster": top_name,
            "confidence": round(float(probs[top]), 4),
            "membership": membership,
        }
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _record_id(song: dict) -> str:
    """Stable per-record id: synthetic records use their repo-relative _path,
    real corpus songs their calibration-cache 'artist/title' id."""
    path = song.get("_path", "")
    if path.startswith("corpus/synthetic_data"):
        return path
    return _cache_id_from_path(path)


def _load_corpus_records() -> list[dict]:
    """The real Indian corpus -- the held-out evaluation pool."""
    from services.tests.conftest import load_corpus
    return load_corpus()


def _load_train_records() -> list[dict]:
    """The training pool: synthetic data + consented seeds (copyright-clean
    contract -- see bayesian_scoring_service.load_train_records; scraped real
    lyrics never train a shipped model). Synthetic tracks' semantic metrics
    come from the persisted synthetic cache (live-fetched once); consented
    seeds live-fetch the same way on first run."""
    from services.bayesian_scoring_service import load_train_records
    return load_train_records()


def evaluate_on_real(bundle: dict) -> dict:
    """Fit statistics of the synthetic-trained GMM on the held-out real
    corpus: average log-likelihood, BIC, and the cluster-assignment table
    (real features standardized with the synthetic-fit scaler)."""
    from collections import Counter
    records = _load_corpus_records()
    X, kept = build_feature_matrix(records, live_fetch=False)
    if not len(X):
        return {"n": 0}
    Xs = bundle["scaler"].transform(X)
    gmm = bundle["gmm"]
    counts = Counter(gmm.predict(Xs).tolist())
    return {
        "n": len(kept),
        "avg_log_likelihood": float(gmm.score(Xs)),
        "bic": float(gmm.bic(Xs)),
        "cluster_counts": {int(c): n for c, n in sorted(counts.items())},
    }


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Train / report / evaluate the GMM style clusters")
    ap.add_argument("--train", action="store_true",
                     help="fit from synthetic data (live-fetching semantic metrics), eval on real corpus")
    ap.add_argument("--report", action="store_true", help="print + persist human cluster labels")
    ap.add_argument("--eval-corpus", action="store_true", help="print each song's cluster")
    ap.add_argument("--compare-kmeans", action="store_true",
                     help="fit GMM + K-means on identical features, compare silhouette scores")
    ap.add_argument("--live-fetch", action="store_true", help="fetch missing semantic metrics from the Space")
    args = ap.parse_args()

    if args.compare_kmeans:
        result = compare_kmeans(live_fetch=args.live_fetch)
        print(f"Compared on {result['n_songs']} songs (silhouette: higher = better-separated clusters, range -1..1)")
        print(f"  GMM              k={result['gmm']['k']:<2} silhouette={result['gmm']['silhouette']:.3f}")
        print(f"  K-means (own k)  k={result['kmeans_own_k']['k']:<2} silhouette={result['kmeans_own_k']['silhouette']:.3f}")
        print(f"  K-means (GMM's k) k={result['kmeans_at_gmm_k']['k']:<2} silhouette={result['kmeans_at_gmm_k']['silhouette']:.3f}")
        return 0

    if args.train:
        # Training pool is synthetic-only; synthetic tracks aren't in the
        # calibration cache, so live-fetch is ON by default for --train
        # (fetched metrics persist to SYNTHETIC_SEMANTIC_CACHE -- only the
        # first run pays the Space calls).
        bundle = train(live_fetch=True)
        bundle["_labels"] = label_clusters(bundle)
        save(bundle)
        print(f"Trained GMM: k={bundle['k']} on {len(bundle['assignments'])} synthetic songs -> {MODEL_PATH}")
        print("\nEvaluating fit on the held-out real corpus...")
        ev = evaluate_on_real(bundle)
        if not ev.get("n"):
            print("  (no real-corpus songs had semantic metrics -- skipped)")
        else:
            print(f"  n={ev['n']}  avg log-likelihood={ev['avg_log_likelihood']:.2f}  BIC={ev['bic']:.0f}")
            print(f"  cluster counts on real corpus: {ev['cluster_counts']}")
        return 0

    if args.report:
        bundle = load()
        labels = bundle.get("_labels") or label_clusters(bundle)
        REPORT_PATH.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
        for c, info in labels.items():
            axes = ", ".join(f"{a}(+{v})" for a, v in info["top_axes"])
            arts = ", ".join(f"{a}×{n}" for a, n in info["dominant_artists"])
            print(f"[cluster {c}] {info['label']}  (n={info['size']})")
            print(f"    axes: {axes}")
            print(f"    artists: {arts}\n")
        print(f"Saved -> {REPORT_PATH}")
        return 0

    if args.eval_corpus:
        bundle = load()
        labels = bundle.get("_labels") or label_clusters(bundle)
        for sid, c in zip(bundle["kept_ids"], bundle["assignments"]):
            print(f"  {labels[str(c)]['label']:32} <- {sid}")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
