"""
prepare_barsnet_dataset.py
--------------------------
Build the BarsNet dataset: line-wise phoneme sequences with <SYL> syllable
boundaries, plus char sequences, element features, tier labels, and source.

Outputs kaggle_dataset/barsnet_dataset.json with records:
  {
    "lines": [[tok_id, ...], ...],   # per-line phoneme ids (BarsNet vocab)
    "chars": [id, ...],              # char ids (first 3072 chars)
    "features": [10 floats],         # V5 explicit element scores /100
    "tier": "elite|mid|commercial" | null (pretrain-only records),
    "source": "real|synthetic|consented",
    "split": "train|val|pretrain"
  }

Vocab: BarsNet specials (PAD=0 UNK=1 MASK=2 LB=3 SYL=4) + G2P phoneme set.
Syllabification: vowel-nucleus rule — a <SYL> boundary opens before each
consonant cluster that precedes a vowel (onset-maximizing enough for rhyme
work; nuclei are what alignment cares about).

Quality rules carried over from prepare_v20_dataset.py:
  dedup'd corpus, 37 label-conflict tracks excluded from FINE-TUNE train
  (they still join PRETRAIN — unlabeled reconstruction is label-agnostic),
  dual-script augmentation, artist-held-out val, same held-out artist list.
"""
import json
import random
import sys
from collections import Counter
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

AXES = ("rhyme", "syllable", "alliteration", "vocabulary", "wordplay",
        "assonance", "consonance", "onomatopoeia", "compound_dens", "holorime_dens")
HELD_OUT_ARTISTS = {"Yungsta", "EPR", "Sikander Kahlon", "Vichaar", "Talha Anjum", "Raga"}
VOWELS = {"a", "aa", "i", "u", "e", "ai", "o", "au", "ri"}

MAX_LINE_TOKENS = 64
MAX_LINES = 64
MAX_CHARS = 3072

# ── BarsNet vocab: specials + G2P phones ─────────────────────────────────────
g2p_vocab = json.loads((NLP_ROOT.parent / "local_real_model" / "vocab_map.json").read_text())
PHONES = sorted(p for p in g2p_vocab["phone2idx"] if not p.startswith("<"))
SPECIALS = ["<PAD>", "<UNK>", "<MASK>", "<LB>", "<SYL>"]
BARS_VOCAB = {t: i for i, t in enumerate(SPECIALS + PHONES)}
SYL_ID = BARS_VOCAB["<SYL>"]

# char vocab: reuse DPST meta's mapping (stable, includes Devanagari)
dpst_meta = json.loads((NLP_ROOT.parent / "local_real_model" / "dpst_model_meta.json").read_text())
CHAR2IDX = dpst_meta["lyric_char2idx"]

print("[*] Loading G2P...")
bundle = load_dpst()
g2p, char2idx_g2p, idx2phone, device = (
    bundle["g2p"], bundle["char2idx"], bundle["idx2phone"], bundle["device"]
)


def syllabify(phones: list[str]) -> list[str]:
    """Insert <SYL> before each onset whose next nucleus follows: boundary
    opens at consonant(s) directly preceding a vowel (skip word-initial)."""
    out: list[str] = []
    for i, p in enumerate(phones):
        if (i > 0 and p not in VOWELS
                and i + 1 < len(phones) and phones[i + 1] in VOWELS):
            out.append("<SYL>")
        out.append(p)
    return out


def line_to_ids(line: str) -> list[int]:
    words = []
    for w in line.split():
        w_clean = "".join(c for c in w if c.isalnum())
        if w_clean:
            words.append(w_clean)
    ids: list[int] = []
    for w in words:
        phones = _translate_word(w, g2p, char2idx_g2p, idx2phone, device)
        for tok in syllabify(phones):
            ids.append(BARS_VOCAB.get(tok, 1))
    return ids[:MAX_LINE_TOKENS]


def encode_lyrics(lyrics: str) -> dict | None:
    lines = []
    for raw in lyrics.split("\n"):
        s = raw.strip()
        if not s or (s.startswith("[") and s.endswith("]")):
            continue
        ids = line_to_ids(s)
        if ids:
            lines.append(ids)
        if len(lines) >= MAX_LINES:
            break
    if len(lines) < 2:
        return None
    chars = [CHAR2IDX.get(c, 1) for c in lyrics.lower()[:MAX_CHARS]]
    return {"lines": lines, "chars": chars}


def romanize(text: str) -> str:
    return "\n".join(" ".join(
        devanagari_to_roman(w) if any(is_devanagari_char(c) for c in w) else w
        for w in line.split(" ")) for line in text.split("\n"))


def has_devanagari(text: str) -> bool:
    return sum(is_devanagari_char(c) for c in text) / max(1, len(text)) > 0.05


# ── Compile raw records (same sourcing as v20 prep) ──────────────────────────
conflicts = json.loads((NLP_ROOT / "corpus" / "data" / "_label_conflict_review.json").read_text(encoding="utf-8"))
excluded = {(c["artist"], c["title"]) for c in conflicts}

raw = []
synth_base = NLP_ROOT / "corpus" / "synthetic_data"
for tier in ("elite", "mid", "commercial"):
    tp = synth_base / tier
    if not tp.exists():
        continue
    for lang_dir in tp.iterdir():
        if not lang_dir.is_dir():
            continue
        for file in lang_dir.glob("*.json"):
            if file.name.startswith("_"):
                continue
            try:
                content = json.loads(file.read_text(encoding="utf-8"))
            except Exception:
                continue
            ly = content.get("lyrics") or content.get("text")
            if ly:
                split = "val" if random.random() < 0.15 else "train"
                raw.append({"lyrics": ly, "tier": tier, "source": "synthetic", "split": split})

for seed in _load_consented_seed_records():
    if seed.get("lyrics") and seed.get("tier"):
        raw.append({"lyrics": seed["lyrics"], "tier": seed["tier"],
                    "source": "consented", "split": "train"})

for rec in load_indian_corpus_records():
    a, t = rec.get("artist"), rec.get("title", "")
    ly = rec.get("lyrics")
    if not ly:
        continue
    if (a, t) in excluded:
        # label unusable, text still valuable: pretrain-only
        raw.append({"lyrics": ly, "tier": None, "source": "real", "split": "pretrain"})
        continue
    split = "val" if a in HELD_OUT_ARTISTS else "train"
    raw.append({"lyrics": ly, "tier": rec["tier"], "source": "real", "split": split})

print(f"[*] Raw records: {len(raw)} ({Counter(r['split'] for r in raw)})")

# ── Encode ───────────────────────────────────────────────────────────────────
out = []
for rec in tqdm(raw, desc="Encoding"):
    variants = [rec["lyrics"]]
    if has_devanagari(rec["lyrics"]):
        variants.append(romanize(rec["lyrics"]))
    for v in variants:
        enc = encode_lyrics(v)
        if enc is None:
            continue
        try:
            sig = _axis_scores_from_lyrics(v)
            feats = [round(float(sig.get(ax, 0.0)) / 100.0, 4) for ax in AXES]
        except Exception:
            feats = [0.0] * len(AXES)
        out.append({**enc, "features": feats, "tier": rec["tier"],
                    "source": rec["source"], "split": rec["split"]})

meta = {
    "bars_vocab": BARS_VOCAB,
    "char2idx": CHAR2IDX,
    "max_line_tokens": MAX_LINE_TOKENS,
    "max_lines": MAX_LINES,
    "max_chars": MAX_CHARS,
    "axes": list(AXES),
    "held_out_artists": sorted(HELD_OUT_ARTISTS),
}

out_dir = NLP_ROOT.parent / "kaggle_dataset"
(out_dir / "barsnet_dataset.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
(out_dir / "barsnet_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

print(f"[+] {len(out)} records -> barsnet_dataset.json")
print("    split×source:", Counter((r["split"], r["source"]) for r in out))
print("    tiers (labeled):", Counter(r["tier"] for r in out if r["tier"]))
print(f"    vocab: {len(BARS_VOCAB)} tokens; meta -> barsnet_meta.json")
