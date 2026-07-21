"""
prepare_v20_dataset.py
----------------------
Build the v20 DPST training dataset with strict data-quality rules:

1. DEDUP: corpus/data has been deduped (see corpus/data/_dedup_log.json).
2. LABEL CONFLICTS: the 37 tracks in corpus/data/_label_conflict_review.json
   (axis profile contradicts artist-tier label) are EXCLUDED from training
   entirely — they stay in the corpus for evaluation only.
3. DUAL-SCRIPT AUGMENTATION: every record containing Devanagari is added a
   second time in machine-romanized form with the SAME label, so script
   carries zero tier signal (kills the char-tower orthography shortcut
   proven by the Emiway M4 A/B test: same lyrics, elite 0.46 dev vs 0.19 roman).
4. ARTIST-HELD-OUT VAL SPLIT: validation real tracks come from whole
   held-out artists (never seen in training), so val accuracy measures
   generalization, not artist memorization. Synthetic val is a random 15%.
   Records carry a "split" field; the Kaggle notebook must respect it.
5. Snippet augmentation (first 8 content lines) applied to TRAIN records only.

Output: kaggle_dataset/dhh_lyrics_dataset.json
"""
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

NLP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NLP_ROOT))

from services.bayesian_scoring_service import (  # noqa: E402
    _axis_scores_from_lyrics,
    _load_consented_seed_records,
    load_indian_corpus_records,
)
from services.dpst_quality_service import _translate_word, load as load_dpst  # noqa: E402
from services.language_utils import devanagari_to_roman, is_devanagari_char  # noqa: E402

random.seed(42)

AXES = (
    "rhyme", "syllable", "alliteration", "vocabulary", "wordplay",
    "assonance", "consonance", "onomatopoeia", "compound_dens", "holorime_dens"
)

# Whole artists held out of training; their tracks form the real-data val split.
HELD_OUT_ARTISTS = {
    "Yungsta",           # elite  (9 tracks)
    "EPR",               # elite  (8)
    "Sikander Kahlon",   # mid    (7)
    "Vichaar",           # mid    (8)
    "Talha Anjum",       # commercial (6)
    "Raga",              # commercial (14)
}

# ── Load exclusion list (label conflicts) ─────────────────────────────────────
conflict_path = NLP_ROOT / "corpus" / "data" / "_label_conflict_review.json"
conflicts = json.loads(conflict_path.read_text(encoding="utf-8"))
excluded_keys = {(c["artist"], c["title"]) for c in conflicts}
print(f"[*] {len(excluded_keys)} label-conflict tracks will be excluded from training")

# ── G2P bundle ────────────────────────────────────────────────────────────────
print("[*] Loading G2P model...")
bundle = load_dpst()
g2p, char2idx, idx2phone, device = (
    bundle["g2p"], bundle["char2idx"], bundle["idx2phone"], bundle["device"]
)


def romanize(text: str) -> str:
    out = []
    for line in text.split("\n"):
        out.append(" ".join(
            devanagari_to_roman(w) if any(is_devanagari_char(c) for c in w) else w
            for w in line.split(" ")
        ))
    return "\n".join(out)


def has_devanagari(text: str) -> bool:
    dev = sum(is_devanagari_char(c) for c in text)
    return dev / max(1, len(text)) > 0.05


def process_text(lyrics_text: str):
    words = []
    for line in lyrics_text.split("\n"):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            continue
        for w in s.split():
            w_clean = "".join(c for c in w if c.isalnum())
            if w_clean:
                words.append(w_clean)
    phone_seq = []
    for w in words[:120]:
        phone_seq.extend(_translate_word(w, g2p, char2idx, idx2phone, device))
    try:
        sig = _axis_scores_from_lyrics(lyrics_text)
        features = [float(sig.get(ax, 0.0)) / 100.0 for ax in AXES]
    except Exception:
        features = [0.0] * len(AXES)
    return phone_seq, features


# ── Compile raw records ───────────────────────────────────────────────────────
raw: list[dict] = []

# A. Synthetic corpus
synth_base = NLP_ROOT / "corpus" / "synthetic_data"
n_synth = 0
for tier in ("elite", "mid", "commercial"):
    tier_path = synth_base / tier
    if not tier_path.exists():
        continue
    for lang_dir in tier_path.iterdir():
        if not lang_dir.is_dir():
            continue
        for file in lang_dir.glob("*.json"):
            if file.name.startswith("_"):
                continue
            try:
                content = json.loads(file.read_text(encoding="utf-8"))
            except Exception:
                continue
            lyrics = content.get("lyrics") or content.get("text")
            if lyrics:
                split = "val" if random.random() < 0.15 else "train"
                raw.append({"lyrics": lyrics, "tier": tier, "artist": None,
                            "source": "synthetic", "split": split})
                n_synth += 1
print(f"[*] Synthetic records: {n_synth}")

# B. Consented seeds (always train)
n_seed = 0
for seed in _load_consented_seed_records():
    if seed.get("lyrics") and seed.get("tier"):
        raw.append({"lyrics": seed["lyrics"], "tier": seed["tier"], "artist": None,
                    "source": "consented", "split": "train"})
        n_seed += 1
print(f"[*] Consented seed records: {n_seed}")

# C. Real Indian corpus (deduped), minus label conflicts, artist-held-out split
n_real, n_excl, n_val = 0, 0, 0
for rec in load_indian_corpus_records():
    artist, title = rec.get("artist"), rec.get("title", "")
    if (artist, title) in excluded_keys:
        n_excl += 1
        continue
    split = "val" if artist in HELD_OUT_ARTISTS else "train"
    if split == "val":
        n_val += 1
    raw.append({"lyrics": rec["lyrics"], "tier": rec["tier"], "artist": artist,
                "source": "real", "split": split})
    n_real += 1
print(f"[*] Real records: {n_real} (excluded {n_excl} label conflicts; "
      f"{n_val} in artist-held-out val)")

# ── Preprocess + augment ──────────────────────────────────────────────────────
out_records: list[dict] = []
stats = {"dual_script": 0, "snippets": 0}

for rec in tqdm(raw, desc="Preprocessing"):
    lyrics, tier, split = rec["lyrics"], rec["tier"], rec["split"]
    variants = [lyrics]

    # Dual-script: add romanized copy of Devanagari text (same label, same split)
    if has_devanagari(lyrics):
        variants.append(romanize(lyrics))
        stats["dual_script"] += 1

    for v in variants:
        phone_seq, features = process_text(v)
        out_records.append({"lyrics": v, "phoneme_seq": phone_seq,
                            "features": features, "tier": tier, "split": split,
                            "source": rec["source"]})

        # Snippet augmentation — TRAIN only (val must stay clean full songs)
        if split == "train":
            lines = [l.strip() for l in v.split("\n")
                     if l.strip() and not (l.strip().startswith("[") and l.strip().endswith("]"))]
            if len(lines) >= 8:
                snip = "\n".join(lines[:8])
                ps, fs = process_text(snip)
                out_records.append({"lyrics": snip, "phoneme_seq": ps,
                                    "features": fs, "tier": tier, "split": split,
                                    "source": rec["source"]})
                stats["snippets"] += 1

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = NLP_ROOT.parent / "kaggle_dataset"
out_dir.mkdir(exist_ok=True)
out_file = out_dir / "dhh_lyrics_dataset.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(out_records, f, ensure_ascii=False)

n_train = sum(r["split"] == "train" for r in out_records)
n_val = sum(r["split"] == "val" for r in out_records)
from collections import Counter
tier_split = Counter((r["split"], r["tier"]) for r in out_records)
print(f"\n[+] Saved {len(out_records)} records -> {out_file}")
print(f"    train={n_train}  val={n_val}")
print(f"    dual-script copies added: {stats['dual_script']}, snippets: {stats['snippets']}")
for s in ("train", "val"):
    print(f"    {s}: " + ", ".join(f"{t}={tier_split[(s, t)]}" for t in ("elite", "mid", "commercial")))
