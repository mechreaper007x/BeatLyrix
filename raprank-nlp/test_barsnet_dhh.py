"""Eval BarsNet on the real DHH corpus (mirrors test_dpst_dhh.py)."""
import glob
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

NLP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NLP_ROOT))

from corpus.real_corpus.indian_tiers import artist_tier_map
from services.barsnet import BarsNet
from services.bayesian_scoring_service import _axis_scores_from_lyrics
from services.dpst_quality_service import _translate_word, load as load_dpst

MODEL_DIR = NLP_ROOT.parent / "local_real_model"
META = json.loads((NLP_ROOT.parent / "kaggle_dataset" / "barsnet_meta.json").read_text(encoding="utf-8"))
BARS_VOCAB = META["bars_vocab"]
CHAR2IDX = META["char2idx"]
MAX_CHARS = META["max_chars"]
AXES = tuple(META["axes"])
VOWELS = {"a", "aa", "i", "u", "e", "ai", "o", "au", "ri"}

_g2p_bundle = load_dpst()


def _syllabify(phones):
    out = []
    for i, p in enumerate(phones):
        if (i > 0 and p not in VOWELS
                and i + 1 < len(phones) and phones[i + 1] in VOWELS):
            out.append("<SYL>")
        out.append(p)
    return out


def encode_lyrics(lyrics: str):
    b = _g2p_bundle
    lines = []
    for raw in lyrics.split("\n"):
        s = raw.strip()
        if not s or (s.startswith("[") and s.endswith("]")):
            continue
        ids = []
        for w in s.split():
            w_clean = "".join(c for c in w if c.isalnum())
            if not w_clean:
                continue
            phones = _translate_word(w_clean, b["g2p"], b["char2idx"], b["idx2phone"], b["device"])
            for tok in _syllabify(phones):
                ids.append(BARS_VOCAB.get(tok, 1))
        if ids:
            lines.append(ids[:META["max_line_tokens"]])
        if len(lines) >= META["max_lines"]:
            break
    if len(lines) < 2:
        return None
    chars = [CHAR2IDX.get(c, 1) for c in lyrics.lower()[:MAX_CHARS]]
    return {"lines": lines, "chars": chars}
TIERS = ("elite", "mid", "commercial")
IDX2TIER = {0: "elite", 1: "mid", 2: "commercial"}
HELD_OUT = {"Yungsta", "EPR", "Sikander Kahlon", "Vichaar", "Talha Anjum", "Raga"}

device = torch.device("cpu")
model = BarsNet(phone_vocab_size=len(BARS_VOCAB), char_vocab_size=len(CHAR2IDX) + 1)
state = torch.load(MODEL_DIR / "barsnet.pt", map_location=device)
model.load_state_dict(state)
model.eval()
print("BarsNet loaded:", sum(p.numel() for p in model.parameters()), "params")

MAXT, MAXL = 64, 64


def predict(lyrics: str) -> str | None:
    enc = encode_lyrics(lyrics)
    if enc is None:
        return None
    lines = torch.zeros(1, MAXL, MAXT, dtype=torch.long)
    for li, ids in enumerate(enc["lines"][:MAXL]):
        t = torch.tensor(ids[:MAXT], dtype=torch.long)
        lines[0, li, :len(t)] = t
    chars = torch.zeros(1, MAX_CHARS, dtype=torch.long)
    c = torch.tensor(enc["chars"][:MAX_CHARS], dtype=torch.long)
    chars[0, :len(c)] = c
    sig = _axis_scores_from_lyrics(lyrics)
    feats = torch.tensor([[float(sig.get(ax, 0.0)) / 100.0 for ax in AXES]], dtype=torch.float)
    with torch.no_grad():
        tier_logits, _, _ = model(lines, chars, feats)
    return IDX2TIER[int(tier_logits.argmax(-1))]


tier_map = artist_tier_map()
results, held = [], defaultdict(list)
t0 = time.time()
files = sorted(glob.glob(str(NLP_ROOT / "corpus" / "data" / "*" / "*.json")))
for f in files:
    if Path(f).name.startswith("_"):
        continue
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    tier = tier_map.get(d.get("artist"))
    if not tier or not d.get("lyrics", "").strip():
        continue
    pred = predict(d["lyrics"])
    if pred is None:
        continue
    results.append((tier, pred))
    if d["artist"] in HELD_OUT:
        held[d["artist"]].append((tier, pred))

n = len(results)
acc = sum(t == p for t, p in results) / n
cm = Counter(results)
print(f"\n== BarsNet | REAL DHH CORPUS: {n} tracks, accuracy {acc:.1%} ({time.time()-t0:.0f}s) ==")
for t in TIERS:
    row = sum(cm[(t, p)] for p in TIERS)
    if row:
        print(f"  {t:<11} " + " ".join(f"{p}={cm[(t,p)]}" for p in TIERS)
              + f"  recall {cm[(t,t)]/row:.1%}")
print("pred dist:", dict(Counter(p for _, p in results)))

print("\nheld-out artists (never in tier training):")
for a in sorted(held):
    rows = held[a]
    print(f"  {a:<16} {rows[0][0]:<11} acc {sum(t==p for t,p in rows)/len(rows):.0%} (n={len(rows)})")
