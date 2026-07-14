"""
Per-element unsupervised clustering: unlike services/gmm_style_service.py
(one holistic GMM over all 26 axes -- "what KIND of rapper"), this fits one
clusterer PER rap-element family -- "what KIND of rhymer", "what KIND of
wordplay writer", etc. Each family gets the algorithm that fits its
dimensionality/distribution shape:

  rhyme    (6 correlated sub-axes, partly zero-inflated)  -> hierarchical
  wordplay (5 correlated sub-axes, partly zero-inflated)  -> hierarchical
  texture  (3 smooth continuous scalars)                  -> GMM (BIC k)
  rare     (3 mostly-zero axes with a rare high tail)      -> HDBSCAN

All features come from corpus/analysis/signature.py::signature(), which is
lyrics-only (no semantic Space, no audio) -- so the training pool is the
full real corpus (tests/conftest.py::load_corpus()) PLUS the consented
seeds (tests/conftest.py::load_consented_seeds()), with no dropout.

total_score is NOT affected -- like gmm_style_service.py, this is descriptive
only. Flow/prosody (needs audio) and semantic axes (needs the semantic Space)
are out of scope here; see gmm_style_service.py for how those dependencies
are normally wired in, if this ever gets extended to cover them.

Usage:
    python -m services.element_cluster_service --train [--family NAME]
    python -m services.element_cluster_service --report [--family NAME]
    python -m services.element_cluster_service --eval-corpus [--family NAME]
    python -m services.element_cluster_service --table [--csv PATH]
    python -m services.element_cluster_service --artist-summary alliteration [--csv PATH]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "corpus" / "synthetic_data"

# k range for hierarchical/GMM families. Capped at 6 for the same reason as
# gmm_style_service.py's BIC_K_RANGE: on a few hundred songs, letting search
# go higher just carves off micro-clusters around single-feature outliers.
K_RANGE = range(2, 7)

FAMILIES: dict[str, dict] = {
    "rhyme": {
        "axes": ("rhyme", "internal", "chain", "multi_dens", "compound_dens", "holorime_dens"),
        "algo": "hierarchical",
    },
    "wordplay": {
        "axes": ("wordplay", "simile", "metaphor", "pun", "entendre"),
        "algo": "hierarchical",
    },
    "texture": {
        "axes": ("alliteration", "assonance", "consonance"),
        "algo": "gmm",
    },
    "rare": {
        "axes": ("onomatopoeia", "compound_dens", "holorime_dens"),
        "algo": "hdbscan",
    },
}

AXIS_DESCRIPTORS: dict[str, str] = {
    "rhyme": "Rhyme-Heavy",
    "internal": "Internal-Rhyme-Rich",
    "chain": "Rhyme-Chain-Heavy",
    "multi_dens": "Multi-Rhyme-Dense",
    "compound_dens": "Compound-Rhyme-Dense",
    "holorime_dens": "Holorime-Dense",
    "wordplay": "Wordplay-Dense",
    "simile": "Simile-Heavy",
    "metaphor": "Metaphor-Heavy",
    "pun": "Pun-Heavy",
    "entendre": "Double-Entendre-Heavy",
    "alliteration": "Alliterative",
    "assonance": "Assonant",
    "consonance": "Consonant-Rich",
    "onomatopoeia": "Onomatopoeic",
}


def _model_path(family: str) -> Path:
    return MODEL_DIR / f"_element_cluster_{family}.pkl"


def _report_path(family: str) -> Path:
    return MODEL_DIR / f"_element_cluster_{family}_report.json"


# ── Pool + features ──────────────────────────────────────────────────────────
def _load_pool() -> list[dict]:
    from services.tests.conftest import load_corpus, load_consented_seeds
    return load_corpus() + load_consented_seeds()


def _family_features(lyrics: str, axes: tuple[str, ...]) -> list[float]:
    from corpus.analysis.signature import signature
    sig = signature(lyrics)
    return [float(sig.get(ax, 0.0)) for ax in axes]


def build_feature_matrix(records: list[dict], axes: tuple[str, ...]):
    """No dropout -- signature() is lyrics-only, every record with lyrics keeps."""
    import numpy as np
    rows, kept = [], []
    for song in records:
        lyrics = song.get("lyrics")
        if not lyrics:
            continue
        rows.append(_family_features(lyrics, axes))
        kept.append(song)
    return np.asarray(rows, dtype=float), kept


# ── Train ─────────────────────────────────────────────────────────────────────
def train_family(family: str, records: list[dict] | None = None) -> dict:
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    cfg = FAMILIES[family]
    axes, algo = cfg["axes"], cfg["algo"]

    if records is None:
        records = _load_pool()
    X, kept = build_feature_matrix(records, axes)
    if len(X) < max(K_RANGE) * 3:
        raise RuntimeError(
            f"Only {len(X)} songs with lyrics for family '{family}' -- need more corpus/consented data."
        )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if algo == "gmm":
        model, k, assignments = _fit_gmm(Xs)
        noise_frac = None
    elif algo == "hierarchical":
        model, k, assignments = _fit_hierarchical(Xs)
        noise_frac = None
    elif algo == "hdbscan":
        model, k, assignments, noise_frac = _fit_hdbscan(Xs)
    else:
        raise ValueError(f"Unknown algo '{algo}' for family '{family}'")

    return {
        "family": family,
        "algo": algo,
        "axes": list(axes),
        "scaler": scaler,
        "model": model,
        "k": k,
        "assignments": assignments,
        "noise_frac": noise_frac,
        "Xs": Xs,  # standardized features, kept for centroid-based labelling
        "kept_ids": [s.get("_path", "?") for s in kept],
        "kept_artists": [s.get("artist", "?") for s in kept],
    }


def _fit_gmm(Xs):
    from sklearn.mixture import GaussianMixture
    best = None
    for k in K_RANGE:
        gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=0, n_init=5, max_iter=300).fit(Xs)
        bic = gm.bic(Xs)
        if best is None or bic < best[0]:
            best = (bic, k, gm)
    _, k, gmm = best
    return gmm, k, gmm.predict(Xs).tolist()


def _fit_hierarchical(Xs):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    best = None
    for k in K_RANGE:
        ac = AgglomerativeClustering(n_clusters=k).fit(Xs)
        sil = silhouette_score(Xs, ac.labels_)
        if best is None or sil > best[0]:
            best = (sil, k, ac)
    _, k, ac = best
    return ac, k, ac.labels_.tolist()


def _fit_hdbscan(Xs):
    from sklearn.cluster import HDBSCAN
    min_cluster_size = max(5, round(0.03 * len(Xs)))
    hdb = HDBSCAN(min_cluster_size=min_cluster_size).fit(Xs)
    labels = hdb.labels_.tolist()
    k = len({l for l in labels if l != -1})
    noise_frac = labels.count(-1) / len(labels) if labels else 0.0
    return hdb, k, labels, noise_frac


def save(bundle: dict) -> Path:
    path = _model_path(bundle["family"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return path


def load(family: str) -> dict:
    with open(_model_path(family), "rb") as f:
        return pickle.load(f)


# ── Labelling ─────────────────────────────────────────────────────────────────
def label_clusters(bundle: dict) -> dict:
    """Per cluster: centroid computed directly from assigned standardized
    rows (works for hierarchical/GMM/HDBSCAN alike, since not every algo
    exposes a `.means_`), top distinguishing axes, dominant artists. HDBSCAN
    noise points (-1) are reported separately, not folded into a cluster.

    Labels only come from axes the cluster is genuinely ELEVATED on (centroid
    > LABEL_THRESHOLD std devs above the corpus mean). Without this, a big
    "average" cluster can get labelled after a rare axis (e.g. holorime_dens)
    just because it's the *least negative* of a set of mostly-negative axes,
    not because the cluster is actually holorime-heavy -- misleading since
    that axis is near-zero for almost every song. Clusters with no axis
    clearing the threshold are labelled "Baseline" instead."""
    import numpy as np
    from collections import Counter

    LABEL_THRESHOLD = 0.15  # standardized units above the corpus mean

    Xs = bundle["Xs"]
    axes = bundle["axes"]
    assignments = bundle["assignments"]
    artists = bundle["kept_artists"]

    cluster_ids = sorted({a for a in assignments if a != -1})
    clusters = {}
    for c in cluster_ids:
        idx = [i for i, a in enumerate(assignments) if a == c]
        centroid = Xs[idx].mean(axis=0)
        order = np.argsort(centroid)[::-1]
        top_axes = [(axes[i], round(float(centroid[i]), 2)) for i in order[:3]]
        elevated = [(axes[i], round(float(centroid[i]), 2)) for i in order if centroid[i] > LABEL_THRESHOLD][:2]
        members = [artists[i] for i in idx]
        dom_artists = Counter(members).most_common(4)
        if elevated:
            label = " · ".join(AXIS_DESCRIPTORS.get(ax, ax) for ax, _ in elevated)
        else:
            label = "Baseline"
        # centroid/max_dist (not just derived stats) are kept so score_family()
        # can do nearest-centroid assignment at runtime for algos (hierarchical,
        # HDBSCAN) that don't expose a .predict() for unseen points.
        max_dist = float(np.linalg.norm(Xs[idx] - centroid, axis=1).max()) if idx else 0.0
        clusters[str(c)] = {
            "label": label,
            "size": len(idx),
            "top_axes": top_axes,
            "dominant_artists": dom_artists,
            "centroid": centroid.tolist(),
            "max_dist": max_dist,
        }

    # Disambiguate duplicate labels -- multiple clusters can independently
    # clear no axis past LABEL_THRESHOLD and all fall back to "Baseline".
    # score_family() keys its membership dict by label, so an undetected
    # collision would silently overwrite one cluster's probability with
    # another's (e.g. confidence=1.0 for the winner but membership shows
    # 0.0, since a second same-labelled cluster's near-zero prob wrote last).
    label_counts = Counter(info["label"] for info in clusters.values())
    dup_seen: Counter = Counter()
    for cid in sorted(clusters, key=lambda k: int(k)):
        label = clusters[cid]["label"]
        if label_counts[label] > 1:
            dup_seen[label] += 1
            clusters[cid]["label"] = f"{label} ({dup_seen[label]})"

    result = {"clusters": clusters}
    if bundle.get("noise_frac") is not None:
        noise_idx = [i for i, a in enumerate(assignments) if a == -1]
        result["noise"] = {
            "size": len(noise_idx),
            "fraction": round(bundle["noise_frac"], 4),
            "artists": Counter(artists[i] for i in noise_idx).most_common(4),
        }
    return result


# ── Inference (runtime) ──────────────────────────────────────────────────────
def score_family(lyrics: str, family: str, bundle: dict | None = None) -> dict | None:
    """Soft membership for a verse in one family's clusters. GMM families use
    predict_proba directly; hierarchical/HDBSCAN don't expose a .predict() for
    unseen points, so they fall back to nearest-centroid assignment against the
    centroids label_clusters() computes and persists per cluster. Recomputes
    labels fresh from the bundle (cheap on this corpus size) rather than trusting
    a possibly-stale cached `_labels` that predates the centroid/max_dist fields.
    Returns None on any failure (missing model / no clusters / bad input)."""
    try:
        import numpy as np
        if bundle is None:
            bundle = load(family)
        axes = bundle["axes"]
        vec = np.asarray(_family_features(lyrics, axes), dtype=float).reshape(1, -1)
        Xs = bundle["scaler"].transform(vec)[0]

        clusters = label_clusters(bundle)["clusters"]
        if not clusters:
            return None
        cluster_ids = sorted(clusters, key=lambda k: int(k))

        if bundle["algo"] == "gmm":
            probs = bundle["model"].predict_proba(Xs.reshape(1, -1))[0]
            membership = {clusters[cid]["label"]: round(float(p), 4) for cid, p in zip(cluster_ids, probs)}
            top_i = int(np.argmax(probs))
            top_cid = cluster_ids[top_i]
            return {
                "cluster": clusters[top_cid]["label"],
                "confidence": round(float(probs[top_i]), 4),
                "membership": membership,
            }

        # hierarchical / hdbscan -- nearest-centroid assignment
        dists = np.asarray([
            float(np.linalg.norm(Xs - np.asarray(clusters[cid]["centroid"], dtype=float)))
            for cid in cluster_ids
        ])
        nearest_i = int(np.argmin(dists))
        nearest_cid = cluster_ids[nearest_i]

        if bundle["algo"] == "hdbscan":
            max_dist = clusters[nearest_cid].get("max_dist")
            if max_dist is not None and dists[nearest_i] > max_dist:
                return {"cluster": "noise", "confidence": None, "membership": {}}

        inv = 1.0 / (dists + 1e-6)
        probs = inv / inv.sum()
        membership = {clusters[cid]["label"]: round(float(p), 4) for cid, p in zip(cluster_ids, probs)}
        return {
            "cluster": clusters[nearest_cid]["label"],
            "confidence": round(float(probs[nearest_i]), 4),
            "membership": membership,
        }
    except FileNotFoundError:
        return None
    except Exception:
        return None


def score_all_families(lyrics: str, bundles: dict[str, dict]) -> dict:
    """Score every loaded family, omitting any that fail/return None."""
    result = {}
    for family, bundle in bundles.items():
        scored = score_family(lyrics, family, bundle=bundle)
        if scored:
            result[family] = scored
    return result


def _load_corpus_records() -> list[dict]:
    return _load_pool()


# All raw axes across the four families, in a fixed display order (dedupes
# axes shared by multiple families, e.g. compound_dens/holorime_dens appear
# in both "rhyme" and "rare").
_RAW_AXES: tuple[str, ...] = tuple(dict.fromkeys(
    ax for cfg in FAMILIES.values() for ax in cfg["axes"]
))


# ── Combined per-song table ──────────────────────────────────────────────────
def build_style_table(real_only: bool = True) -> list[dict]:
    """One row per song: artist/title + each family's cluster label + every
    raw axis score (from signature(), so the underlying number behind each
    label is visible, not just the cluster name). Cluster labels are joined
    by the song's _path against the assignments each bundle already stored
    at train time (no recompute); raw scores are recomputed from lyrics
    since signature() doesn't get persisted in the pickle. real_only=True
    excludes the consented seeds (no 'title', not the corpus under test)."""
    from services.tests.conftest import load_corpus
    from corpus.analysis.signature import signature

    bundles = {family: load(family) for family in FAMILIES}
    labels = {family: (b.get("_labels") or label_clusters(b))["clusters"] for family, b in bundles.items()}

    # _path -> {family: cluster_id}
    by_path: dict[str, dict[str, int]] = {}
    for family, b in bundles.items():
        for path, c in zip(b["kept_ids"], b["assignments"]):
            by_path.setdefault(path, {})[family] = c

    songs = load_corpus() if real_only else _load_pool()
    rows = []
    for song in songs:
        path = song.get("_path", "?")
        assign = by_path.get(path)
        if assign is None:
            continue  # dropped from every family (missing lyrics) -- shouldn't happen for real corpus
        row = {"artist": song.get("artist", "?"), "title": song.get("title", path)}
        for family in FAMILIES:
            c = assign.get(family)
            if c is None:
                row[family] = "?"
            elif c == -1:
                row[family] = "noise"
            else:
                row[family] = labels[family][str(c)]["label"]
        sig = signature(song["lyrics"])
        for ax in _RAW_AXES:
            row[ax] = round(float(sig.get(ax, 0.0)), 2)
        rows.append(row)
    return rows


def build_artist_summary(axis: str = "alliteration", real_only: bool = True) -> list[dict]:
    """Per-artist mean/n for one raw axis, sorted highest-mean first."""
    import statistics as st
    from collections import defaultdict
    from services.tests.conftest import load_corpus
    from corpus.analysis.signature import signature

    songs = load_corpus() if real_only else _load_pool()
    by_artist: dict[str, list[float]] = defaultdict(list)
    for song in songs:
        sig = signature(song["lyrics"])
        by_artist[song.get("artist", "?")].append(float(sig.get(axis, 0.0)))

    rows = [{"artist": a, "n": len(v), f"mean_{axis}": round(st.mean(v), 2)} for a, v in by_artist.items()]
    rows.sort(key=lambda r: -r[f"mean_{axis}"])
    return rows


def _table_columns(rows: list[dict]) -> list[str]:
    if not rows:
        return ["artist", "title"] + sorted(FAMILIES)
    fixed = [c for c in ("artist", "title") if c in rows[0]]
    return fixed + [c for c in rows[0] if c not in fixed]


def _print_table(rows: list[dict]) -> None:
    cols = _table_columns(rows)
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) if rows else len(c) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def _write_csv(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = _table_columns(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Train / report / evaluate per-element clusters")
    ap.add_argument("--train", action="store_true", help="fit from corpus+consented pool and pickle")
    ap.add_argument("--report", action="store_true", help="print + persist human cluster labels")
    ap.add_argument("--eval-corpus", action="store_true", help="print each song's cluster")
    ap.add_argument("--table", action="store_true",
                     help="print a combined artist/title x family cluster table for the real corpus")
    ap.add_argument("--csv", type=Path, help="with --table/--artist-summary, also write to this CSV path")
    ap.add_argument("--artist-summary", metavar="AXIS",
                     help="print per-artist mean for one raw axis (e.g. alliteration), sorted highest first")
    ap.add_argument("--family", choices=sorted(FAMILIES), help="restrict to one family (default: all)")
    args = ap.parse_args()

    families = [args.family] if args.family else sorted(FAMILIES)

    if args.train:
        pool = _load_pool()
        for family in families:
            bundle = train_family(family, records=pool)
            bundle["_labels"] = label_clusters(bundle)
            path = save(bundle)
            extra = f" noise={bundle['noise_frac']:.1%}" if bundle["noise_frac"] is not None else ""
            print(f"[{family}] {bundle['algo']}: k={bundle['k']} on {len(bundle['assignments'])} songs{extra} -> {path}")
        return 0

    if args.report:
        for family in families:
            bundle = load(family)
            labels = bundle.get("_labels") or label_clusters(bundle)
            _report_path(family).write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"=== {family} ({bundle['algo']}) ===")
            for c, info in labels["clusters"].items():
                axes_s = ", ".join(f"{a}(+{v})" for a, v in info["top_axes"])
                arts = ", ".join(f"{a}×{n}" for a, n in info["dominant_artists"])
                print(f"[cluster {c}] {info['label']}  (n={info['size']})")
                print(f"    axes: {axes_s}")
                print(f"    artists: {arts}")
            if "noise" in labels:
                n = labels["noise"]
                print(f"[noise] n={n['size']} ({n['fraction']:.1%}) artists: {n['artists']}")
            print(f"Saved -> {_report_path(family)}\n")
        return 0

    if args.eval_corpus:
        for family in families:
            bundle = load(family)
            labels = bundle.get("_labels") or label_clusters(bundle)
            print(f"=== {family} ===")
            for sid, c in zip(bundle["kept_ids"], bundle["assignments"]):
                name = "noise" if c == -1 else labels["clusters"][str(c)]["label"]
                print(f"  {name:32} <- {sid}")
        return 0

    if args.table:
        rows = build_style_table(real_only=True)
        _print_table(rows)
        print(f"\n{len(rows)} real corpus songs")
        if args.csv:
            _write_csv(rows, args.csv)
            print(f"Saved -> {args.csv}")
        return 0

    if args.artist_summary:
        rows = build_artist_summary(axis=args.artist_summary, real_only=True)
        _print_table(rows)
        if args.csv:
            _write_csv(rows, args.csv)
            print(f"Saved -> {args.csv}")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
