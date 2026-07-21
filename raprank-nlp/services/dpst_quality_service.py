"""
dpst_quality_service.py
-----------------------
Inference wrapper for the Dual-tower Phonetic-Semantic Transformer (DPST)
classifier trained on Kaggle.

Architecture (must mirror the training notebook exactly):
  Tower A  — PhoneticTransformerEncoder: G2P phoneme sequences → Transformer encoder
  Tower B  — CharCNNEncoder: raw lyrics characters → multi-scale Conv1d
  Fusion   — CrossAttentionFusion: phonetic queries attend to char-semantic keys
  Head     — 3-class classifier → (elite=0, mid=1, commercial=2)

Public API:
    load()                 -> bundle dict  (call once at startup in main.py)
    predict(bundle, lyrics) -> dict with tier, confidence, probabilities
"""
from __future__ import annotations

import json
import math
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Explicit rap-signature axes (must match training preprocessing order) ──────
_SIG_AXES = (
    "rhyme", "syllable", "alliteration", "vocabulary", "wordplay",
    "assonance", "consonance", "onomatopoeia", "compound_dens", "holorime_dens"
)
NUM_EXPLICIT = len(_SIG_AXES)  # 10

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent
_LOCAL_MODEL_DIR = _HERE.parent / "local_real_model"

BARSNET_MODEL_PATH = _LOCAL_MODEL_DIR / "barsnet.pt"
if not BARSNET_MODEL_PATH.exists():
    BARSNET_MODEL_PATH = _HERE.parent / "barsnet.pt"

BARSNET_META_PATH = _HERE.parent / "kaggle_dataset" / "barsnet_meta.json"

DPST_MODEL_PATH = _LOCAL_MODEL_DIR / "dhh_classifier.pt"
DPST_META_PATH  = _LOCAL_MODEL_DIR / "dpst_model_meta.json"
G2P_MODEL_PATH  = _LOCAL_MODEL_DIR / "g2p_model.pt"
G2P_VOCAB_PATH  = _LOCAL_MODEL_DIR / "vocab_map.json"

VOWELS = {"a", "aa", "i", "u", "e", "ai", "o", "au", "ri"}

from services.barsnet import BarsNet  # noqa: E402

# ── Label map ─────────────────────────────────────────────────────────────────
IDX2TIER = {0: "elite", 1: "mid", 2: "commercial"}
TIER2IDX = {v: k for k, v in IDX2TIER.items()}

# ── G2P Model (must mirror Kaggle training notebook) ──────────────────────────
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerG2P(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_token)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_token)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)

    def forward(self, src, tgt, tgt_mask=None, **kwargs):
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(out)


# ── DPST Classifier (must mirror Kaggle training notebook) ───────────────────
CHAR_PAD = 0
CHAR_UNK = 1
D_MODEL  = 256


class CharCNNEncoder(nn.Module):
    def __init__(self, char_vocab_size: int, embed_dim: int = 64, d_model: int = D_MODEL):
        super().__init__()
        self.embed = nn.Embedding(char_vocab_size, embed_dim, padding_idx=CHAR_PAD)
        # All-odd kernel sizes → same output length after padding=k//2
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, d_model // 4, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7, 9]
        ])
        self.proj    = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(char_ids).transpose(1, 2)      # (B, embed, L)
        pooled = [torch.relu(conv(x)) for conv in self.convs]  # each (B, d//4, L)
        out = torch.cat(pooled, dim=1).transpose(1, 2)          # (B, L, d)
        return self.dropout(self.proj(out))


class PhoneticTransformerEncoder(nn.Module):
    def __init__(self, phone_vocab_size: int, d_model: int = D_MODEL):
        super().__init__()
        self.embed = nn.Embedding(phone_vocab_size, d_model, padding_idx=0)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=512,
                dropout=0.1, batch_first=True,
            ),
            num_layers=3,
        )

    def forward(self, phone_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = (phone_ids == 0)
        x = self.embed(phone_ids)
        return self.encoder(x, src_key_padding_mask=padding_mask)


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = D_MODEL, nhead: int = 8):
        super().__init__()
        self.mha     = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, phone_feats: torch.Tensor, char_feats: torch.Tensor, char_padding_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.mha(
            phone_feats, char_feats, char_feats,
            key_padding_mask=char_padding_mask
        )
        return self.norm(phone_feats + self.dropout(attn_out))


class DPSTClassifier(nn.Module):
    """DPST-Hybrid: phonetic tower + char-CNN tower + 10 explicit rap features."""
    def __init__(
        self,
        char_vocab_size: int,
        phone_vocab_size: int,
        num_classes: int = 3,
        d_model: int = D_MODEL,
        num_explicit: int = NUM_EXPLICIT,
    ):
        super().__init__()
        self.tower_a  = PhoneticTransformerEncoder(phone_vocab_size, d_model)
        self.tower_b  = CharCNNEncoder(char_vocab_size, d_model=d_model)
        self.fusion   = CrossAttentionFusion(d_model)
        # Input = d_model (phone_repr) + d_model (char_repr) + num_explicit (10)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2 + num_explicit, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
        # v21 multi-task element head (predicts the 10 axis scores from tower
        # representations). Absent in v19/v20 checkpoints — load() handles both.
        self.aux_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_explicit),
            nn.Sigmoid(),
        )

    def forward(
        self,
        phone_ids: torch.Tensor,
        char_ids: torch.Tensor,
        explicit_feats: torch.Tensor,
    ) -> torch.Tensor:
        phone_feats = self.tower_a(phone_ids)
        char_feats  = self.tower_b(char_ids)

        char_padding_mask = (char_ids == 0)
        fused       = self.fusion(phone_feats, char_feats, char_padding_mask)

        # Masked mean pooling for Tower A
        phone_mask = (phone_ids != 0).unsqueeze(-1).float()
        phone_repr = (fused * phone_mask).sum(dim=1) / (phone_mask.sum(dim=1) + 1e-9)

        # Masked mean pooling for Tower B
        char_mask = (char_ids != 0).unsqueeze(-1).float()
        char_repr = (char_feats * char_mask).sum(dim=1) / (char_mask.sum(dim=1) + 1e-9)

        # Concatenate tower representations with 10 explicit signature features
        towers = torch.cat([phone_repr, char_repr], dim=-1)
        combined = torch.cat([towers, explicit_feats], dim=-1)
        return self.classifier(combined), self.aux_head(towers)


# In-memory global word cache to avoid running sequence-to-sequence G2P for identical words
_G2P_CACHE: dict[str, list[str]] = {}

# ── G2P inference helper ──────────────────────────────────────────────────────
def _translate_word(
    word: str,
    g2p_model: TransformerG2P,
    char2idx: dict,
    idx2phone: dict,
    device: torch.device,
    max_len: int = 15,
) -> list[str]:
    w_lower = word.lower()
    if w_lower in _G2P_CACHE:
        return _G2P_CACHE[w_lower]

    indices = [char2idx.get(c.lower(), UNK_token) for c in word] + [EOS_token]
    src = torch.tensor([indices], dtype=torch.long, device=device)
    tgt_indices = [SOS_token]
    for _ in range(max_len):
        tgt = torch.tensor([tgt_indices], dtype=torch.long, device=device)
        tgt_mask = g2p_model._causal_mask(len(tgt_indices), device)
        with torch.no_grad():
            out = g2p_model(src=src, tgt=tgt, tgt_mask=tgt_mask)
        next_tok = out[0, -1].argmax().item()
        if next_tok == EOS_token:
            break
        tgt_indices.append(next_tok)
    
    phones = [idx2phone[i] for i in tgt_indices[1:] if i in idx2phone]
    _G2P_CACHE[w_lower] = phones
    
    # Save periodically to disk (every 100 new additions)
    if len(_G2P_CACHE) % 100 == 0:
        try:
            cache_path = _LOCAL_MODEL_DIR / "g2p_word_cache.json"
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(_G2P_CACHE, f)
        except Exception:
            pass
    return phones



# ── Public API ────────────────────────────────────────────────────────────────
def _syllabify(phones: list[str]) -> list[str]:
    out = []
    for i, p in enumerate(phones):
        if (i > 0 and p not in VOWELS
                and i + 1 < len(phones) and phones[i + 1] in VOWELS):
            out.append("<SYL>")
        out.append(p)
    return out


def load() -> dict:
    """
    Load G2P model, BarsNet V2 (or legacy DPST) classifier, and metadata into memory.
    Raises on any file-not-found or shape mismatch so main.py can guard it.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load G2P word cache from disk if available
    global _G2P_CACHE
    cache_path = _LOCAL_MODEL_DIR / "g2p_word_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                _G2P_CACHE.update(json.load(f))
            logger.info("DPST: Loaded %d words from G2P disk cache.", len(_G2P_CACHE))
        except Exception as e:
            logger.warning("DPST: Failed to load disk cache: %s", e)

    # Load G2P vocab
    with open(G2P_VOCAB_PATH, "r", encoding="utf-8") as f:
        g2p_vocab = json.load(f)
    char2idx  = g2p_vocab["char2idx"]
    idx2phone = {int(k): v for k, v in g2p_vocab["idx2phone"].items()}
    phone2idx = g2p_vocab["phone2idx"]

    # Load G2P model
    g2p = TransformerG2P(len(char2idx), len(phone2idx)).to(device)
    g2p.load_state_dict(torch.load(G2P_MODEL_PATH, map_location=device))
    g2p.eval()
    logger.info("DPST: G2P model loaded (%d chars, %d phones).", len(char2idx), len(phone2idx))

    # ── Check for BarsNet V2 model ───────────────────────────────────────────
    if BARSNET_MODEL_PATH.exists() and BARSNET_META_PATH.exists():
        try:
            meta = json.loads(BARSNET_META_PATH.read_text(encoding="utf-8"))
            bars_vocab = meta["bars_vocab"]
            bars_char2idx = meta["char2idx"]
            bars_axes = tuple(meta["axes"])
            bars_max_line_tokens = meta.get("max_line_tokens", 64)
            bars_max_lines = meta.get("max_lines", 64)
            bars_max_chars = meta.get("max_chars", 3072)

            barsnet_model = BarsNet(phone_vocab_size=len(bars_vocab), char_vocab_size=len(bars_char2idx) + 1).to(device)
            barsnet_state = torch.load(BARSNET_MODEL_PATH, map_location=device)
            barsnet_model.load_state_dict(barsnet_state)
            barsnet_model.eval()
            logger.info("DPST: BarsNet V2 loaded (%d params).", sum(p.numel() for p in barsnet_model.parameters()))

            return {
                "device": device,
                "is_barsnet": True,
                "has_aux": True,
                "g2p": g2p,
                "char2idx": char2idx,
                "idx2phone": idx2phone,
                "bars_vocab": bars_vocab,
                "bars_char2idx": bars_char2idx,
                "bars_axes": bars_axes,
                "bars_max_line_tokens": bars_max_line_tokens,
                "bars_max_lines": bars_max_lines,
                "bars_max_chars": bars_max_chars,
                "classifier": barsnet_model,
            }
        except Exception as exc:
            logger.warning("DPST: BarsNet V2 loading failed (%s), falling back to DPSTClassifier...", exc)

    # ── Fallback: Legacy DPST Classifier ─────────────────────────────────────
    with open(DPST_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    lyric_char2idx = meta["lyric_char2idx"]
    phoneme_vocab  = meta["phoneme_vocab"]
    max_phone_len  = meta["max_phone_len"]
    max_char_len   = meta["max_char_len"]

    classifier = DPSTClassifier(
        char_vocab_size=meta["char_vocab_size"],
        phone_vocab_size=meta["phone_vocab_size"],
    ).to(device)
    state = torch.load(DPST_MODEL_PATH, map_location=device)
    has_aux = any(k.startswith("aux_head.") for k in state)
    if has_aux:
        classifier.load_state_dict(state)
    else:
        missing, unexpected = classifier.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
    classifier.eval()
    logger.info(
        "DPST: classifier loaded (char_vocab=%d, phone_vocab=%d).",
        meta["char_vocab_size"], meta["phone_vocab_size"],
    )

    return {
        "device": device,
        "is_barsnet": False,
        "has_aux": has_aux,
        "g2p": g2p,
        "char2idx": char2idx,
        "idx2phone": idx2phone,
        "lyric_char2idx": lyric_char2idx,
        "phoneme_vocab": phoneme_vocab,
        "classifier": classifier,
        "max_phone_len": max_phone_len,
        "max_char_len": max_char_len,
    }


def predict(bundle: dict, lyrics: str) -> dict:
    """
    Run BarsNet V2 (or legacy DPST-Hybrid) inference on raw lyrics text.
    """
    if bundle.get("is_barsnet"):
        device = bundle["device"]
        g2p = bundle["g2p"]
        char2idx = bundle["char2idx"]
        idx2phone = bundle["idx2phone"]
        bars_vocab = bundle["bars_vocab"]
        bars_char2idx = bundle["bars_char2idx"]
        bars_axes = bundle["bars_axes"]
        max_lt = bundle["bars_max_line_tokens"]
        max_l = bundle["bars_max_lines"]
        max_c = bundle["bars_max_chars"]
        classifier = bundle["classifier"]

        lines_encoded = []
        for raw in lyrics.split("\n"):
            s = raw.strip()
            if not s or (s.startswith("[") and s.endswith("]")):
                continue
            ids = []
            for w in s.split():
                w_clean = "".join(c for c in w if c.isalnum())
                if not w_clean:
                    continue
                phones = _translate_word(w_clean, g2p, char2idx, idx2phone, device)
                for tok in _syllabify(phones):
                    ids.append(bars_vocab.get(tok, 1))
            if ids:
                lines_encoded.append(ids[:max_lt])
            if len(lines_encoded) >= max_l:
                break

        if len(lines_encoded) < 2:
            lines_encoded = [[1], [1]]

        char_ids = [bars_char2idx.get(c, 1) for c in lyrics.lower()[:max_c]]

        lines_tensor = torch.zeros(1, max_l, max_lt, dtype=torch.long, device=device)
        for li, ids in enumerate(lines_encoded[:max_l]):
            lines_tensor[0, li, :len(ids)] = torch.tensor(ids[:max_lt], dtype=torch.long, device=device)

        chars_tensor = torch.zeros(1, max_c, dtype=torch.long, device=device)
        chars_tensor[0, :len(char_ids)] = torch.tensor(char_ids[:max_c], dtype=torch.long, device=device)

        explicit_vals = [0.0] * len(bars_axes)
        try:
            from services.bayesian_scoring_service import _axis_scores_from_lyrics
            sig = _axis_scores_from_lyrics(lyrics)
            explicit_vals = [float(sig.get(ax, 0.0)) / 100.0 for ax in bars_axes]
        except Exception as e:
            logger.warning("BarsNet V2: Could not compute explicit features: %s", e)

        explicit_tensor = torch.tensor([explicit_vals], dtype=torch.float, device=device)

        with torch.no_grad():
            tier_logits, elem_preds, _ = classifier(lines_tensor, chars_tensor, explicit_tensor)
            probs = torch.softmax(tier_logits, dim=-1)[0].cpu().tolist()

        pred_idx = int(torch.tensor(probs).argmax().item())
        tier = IDX2TIER[pred_idx]
        confidence = round(probs[pred_idx], 4)
        probabilities = {IDX2TIER[i]: round(probs[i], 4) for i in range(3)}

        res = {
            "tier": tier,
            "confidence": confidence,
            "probabilities": probabilities,
        }
        if elem_preds is not None:
            vals = elem_preds[0].cpu().tolist()
            res["predicted_elements"] = {
                ax: round(v * 100.0, 1) for ax, v in zip(bars_axes, vals)
            }
        return res

    # ── Legacy DPST Classifier Branch ────────────────────────────────────────
    device          = bundle["device"]
    g2p             = bundle["g2p"]
    char2idx        = bundle["char2idx"]
    idx2phone       = bundle["idx2phone"]
    lyric_char2idx  = bundle["lyric_char2idx"]
    phoneme_vocab   = bundle["phoneme_vocab"]
    classifier      = bundle["classifier"]
    max_phone_len   = bundle["max_phone_len"]
    max_char_len    = bundle["max_char_len"]

    words = []
    for line in lyrics.split("\n"):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        for w in stripped.split():
            w_clean = "".join(c for c in w if c.isalnum())
            if w_clean:
                words.append(w_clean)

    phone_seq: list[str] = []
    for w in words[:120]:
        phone_seq.extend(_translate_word(w, g2p, char2idx, idx2phone, device))

    phone_indices = [phoneme_vocab.get(p, 1) for p in phone_seq][:max_phone_len]
    if len(phone_indices) < max_phone_len:
        phone_indices += [0] * (max_phone_len - len(phone_indices))

    char_indices = [lyric_char2idx.get(c, CHAR_UNK) for c in lyrics.lower()][:max_char_len]
    if len(char_indices) < max_char_len:
        char_indices += [CHAR_PAD] * (max_char_len - len(char_indices))

    explicit_vals: list[float] = [0.0] * NUM_EXPLICIT
    try:
        from services.bayesian_scoring_service import _axis_scores_from_lyrics
        sig = _axis_scores_from_lyrics(lyrics)
        explicit_vals = [float(sig.get(ax, 0.0)) / 100.0 for ax in _SIG_AXES]
    except Exception as e:
        logger.warning("DPST: Could not compute explicit features, using zeros: %s", e)

    phone_tensor    = torch.tensor([phone_indices],   dtype=torch.long,  device=device)
    char_tensor     = torch.tensor([char_indices],    dtype=torch.long,  device=device)
    explicit_tensor = torch.tensor([explicit_vals],   dtype=torch.float, device=device)

    with torch.no_grad():
        logits, aux_pred = classifier(phone_tensor, char_tensor, explicit_tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    pred_idx      = int(torch.tensor(probs).argmax().item())
    tier          = IDX2TIER[pred_idx]
    confidence    = round(probs[pred_idx], 4)
    probabilities = {IDX2TIER[i]: round(probs[i], 4) for i in range(3)}

    result = {
        "tier": tier,
        "confidence": confidence,
        "probabilities": probabilities,
    }
    if bundle.get("has_aux"):
        vals = aux_pred[0].cpu().tolist()
        result["predicted_elements"] = {
            ax: round(v * 100.0, 1) for ax, v in zip(_SIG_AXES, vals)
        }
    return result

