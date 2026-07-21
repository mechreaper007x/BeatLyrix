"""
BarsNet V2 — a structure-first encoder/decoder for rap lyrics, built from
primitives and enhanced with RoPE, Symmetric Co-Attention, and Focal Loss.

Design (V2 upgraded 2026-07-21):
  * Input: per-line phoneme sequences with <SYL> syllable-boundary tokens.
  * LineEncoder: shared self-attention blocks over each line's phonemes with
    BACKWARD positional encoding + RoPE (Rotary Position Embeddings) — positions
    counted from the line END, so "2nd-last syllable" is the same feature in every
    line (rhyme lives there).
  * RhymeGeometry: L×L cosine-similarity matrix over line-end suffix
    embeddings → 2D Conv with MASKED spatial max/mean pooling → rhyme-pattern vector.
  * Symmetric Co-Attention: Bidirectional attention between Phonetic Tower & CharCNN.
  * SongEncoder: self-attention over line vectors with RoPE positional embeddings.
  * CharCNN branch: multi-kernel orthographic texture branch.
  * Multi-task heads: tier (3-way focal loss), elements (10-dim), source-adversarial
    (gradient reversal).
  * MaskedSpanDecoder: cross-attention Transformer decoder for pretraining.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Special token ids (phoneme vocab is built around these) ──────────────────
PAD, UNK, MASK, LB, SYL = 0, 1, 2, 3, 4
N_SPECIALS = 5

D_MODEL = 192
N_HEADS = 4
FF_MULT = 4
MAX_LINE_TOKENS = 64      # phonemes+<SYL> per line (truncate longer lines)
MAX_LINES = 64            # lines per song
SUFFIX_TOKENS = 8         # line-end tokens used for rhyme geometry
NUM_ELEMENTS = 10


# ── Rotary Position Embedding (RoPE) ─────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    @staticmethod
    def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, h, T, dk)
        d_half = x.shape[-1] // 2
        x1, x2 = x[..., :d_half], x[..., d_half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)


# ── Multi-Head Self-Attention with RoPE ──────────────────────────────────────
class SelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS):
        super().__init__()
        assert d_model % n_heads == 0
        self.dk = d_model // n_heads
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.dk)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)

        cos, sin = self.rope(q, T)
        q = RotaryEmbedding.apply_rope(q, cos, sin)
        k = RotaryEmbedding.apply_rope(k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if pad_mask is not None:
            att = att.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = torch.nan_to_num(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, dropout: float = 0.15):
        super().__init__()
        self.attn = SelfAttentionRoPE(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, FF_MULT * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(FF_MULT * d_model, d_model),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.drop(self.attn(self.n1(x), pad_mask))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


def sinusoid_table(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ── Line encoder with BACKWARD positional encoding ───────────────────────────
class LineEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = D_MODEL, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.register_buffer("pe", sinusoid_table(MAX_LINE_TOKENS, d_model))
        self.blocks = nn.ModuleList([EncoderBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, line_tokens: torch.Tensor):
        B, L, T = line_tokens.shape
        flat = line_tokens.view(B * L, T)
        pad = flat == PAD
        lengths = (~pad).sum(dim=1)
        idx = torch.arange(T, device=flat.device).unsqueeze(0)
        back = (lengths.unsqueeze(1) - 1 - idx).clamp(min=0)
        x = self.embed(flat) + self.pe[back]
        for blk in self.blocks:
            x = blk(x, pad)
        x = self.norm(x)

        m = (~pad).unsqueeze(-1).float()
        line_vec = (x * m).sum(1) / (m.sum(1) + 1e-9)

        suf_mask = ((back < SUFFIX_TOKENS) & ~pad).unsqueeze(-1).float()
        suffix = (x * suf_mask).sum(1) / (suf_mask.sum(1) + 1e-9)

        empty = (lengths == 0)
        line_vec = line_vec.masked_fill(empty.unsqueeze(-1), 0.0)
        suffix = suffix.masked_fill(empty.unsqueeze(-1), 0.0)
        return (x.view(B, L, T, -1),
                line_vec.view(B, L, -1),
                suffix.view(B, L, -1),
                empty.view(B, L))


# ── Rhyme geometry with Masked Spatial Pooling ────────────────────────────────
class RhymeGeometry(nn.Module):
    def __init__(self, d_model: int = D_MODEL, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.BatchNorm2d(8), nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.GELU(),
        )
        self.proj = nn.Linear(16 * 2, out_dim)

    def forward(self, suffix: torch.Tensor, line_empty: torch.Tensor):
        s = F.normalize(suffix, dim=-1)
        sim = s @ s.transpose(1, 2)
        valid = (~line_empty).float()
        vmask = valid.unsqueeze(2) * valid.unsqueeze(1)
        sim = sim * vmask
        eye = torch.eye(sim.size(1), device=sim.device).unsqueeze(0)
        sim = sim * (1.0 - eye)
        
        f = self.conv(sim.unsqueeze(1)) # (B, 16, L, L)
        
        # Mask out padding entries before spatial max-pooling to avoid conv bias leaking
        f_masked = f.masked_fill(~vmask.unsqueeze(1).bool(), float("-inf"))
        denom = vmask.sum(dim=(1, 2)).clamp(min=1.0)
        
        mean_pool = (f * vmask.unsqueeze(1)).sum(dim=(2, 3)) / denom.unsqueeze(1)
        max_pool = torch.nan_to_num(f_masked.amax(dim=(2, 3)))
        return self.proj(torch.cat([mean_pool, max_pool], dim=-1))


# ── Symmetric Dual-Tower Co-Attention ────────────────────────────────────────
class SymmetricCoAttention(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS):
        super().__init__()
        self.mha_phone = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mha_char = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_phone = nn.LayerNorm(d_model)
        self.norm_char = nn.LayerNorm(d_model)

    def forward(self, song_vecs: torch.Tensor, char_vecs: torch.Tensor):
        # song_vecs: (B, L, D), char_vecs: (B, C, D)
        p_out, _ = self.mha_phone(song_vecs, char_vecs, char_vecs)
        c_out, _ = self.mha_char(char_vecs, song_vecs, song_vecs)
        song_fused = self.norm_phone(song_vecs + p_out)
        char_fused = self.norm_char(char_vecs + c_out)
        return song_fused, char_fused


# ── Song encoder over line vectors ───────────────────────────────────────────
class SongEncoder(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_layers: int = 2):
        super().__init__()
        self.register_buffer("pe", sinusoid_table(MAX_LINES, d_model))
        self.blocks = nn.ModuleList([EncoderBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, line_vecs: torch.Tensor, line_empty: torch.Tensor):
        B, L, D = line_vecs.shape
        x = line_vecs + self.pe[:L].unsqueeze(0)
        for blk in self.blocks:
            x = blk(x, line_empty)
        x = self.norm(x)
        return x # Return sequence of line vectors for co-attention


# ── Char branch (texture) ────────────────────────────────────────────────────
class CharCNN(nn.Module):
    def __init__(self, char_vocab: int, embed_dim: int = 48, out_dim: int = D_MODEL):
        super().__init__()
        self.embed = nn.Embedding(char_vocab, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, out_dim // 4, kernel_size=k, padding=k // 2)
            for k in (3, 5, 7, 9)
        ])
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, char_ids: torch.Tensor):
        x = self.embed(char_ids).transpose(1, 2)
        feats = torch.cat([F.gelu(c(x)) for c in self.convs], dim=1).transpose(1, 2)
        return self.norm(feats) # (B, C, D)


# ── Gradient reversal for the source-adversarial head ────────────────────────
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def grad_reverse(x, lamb: float = 1.0):
    return _GradReverse.apply(x, lamb)


# ── Focal Loss for Class Imbalance & Overconfidence Prevention ───────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sample_weight: torch.Tensor | None = None):
        num_classes = logits.size(-1)
        log_preds = F.log_softmax(logits, dim=-1)
        preds = log_preds.exp()
        
        with torch.no_grad():
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
            if self.label_smoothing > 0:
                one_hot = (1.0 - self.label_smoothing) * one_hot + self.label_smoothing / num_classes

        p_t = (preds * one_hot).sum(dim=-1)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = - (one_hot * log_preds).sum(dim=-1) * focal_weight

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = loss * alpha_t

        if sample_weight is not None:
            return (loss * sample_weight).sum() / sample_weight.sum().clamp(min=1.0)
        return loss.mean()


# ── The full model ───────────────────────────────────────────────────────────
class BarsNet(nn.Module):
    def __init__(self, phone_vocab_size: int, char_vocab_size: int,
                 d_model: int = D_MODEL, rhyme_dim: int = 64,
                 num_classes: int = 3, num_elements: int = NUM_ELEMENTS):
        super().__init__()
        self.line_enc = LineEncoder(phone_vocab_size, d_model)
        self.rhyme = RhymeGeometry(d_model, rhyme_dim)
        self.song_enc = SongEncoder(d_model)
        self.char_cnn = CharCNN(char_vocab_size, out_dim=d_model)
        self.co_attn = SymmetricCoAttention(d_model, n_heads=N_HEADS)
        
        self.repr_norm = nn.LayerNorm(d_model * 2 + rhyme_dim)

        rep = d_model * 2 + rhyme_dim
        self.tier_head = nn.Sequential(
            nn.Linear(rep + num_elements, 192), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(192, 64), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )
        self.element_head = nn.Sequential(
            nn.Linear(rep, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_elements), nn.Sigmoid(),
        )
        self.source_head = nn.Sequential(
            nn.Linear(rep, 64), nn.GELU(), nn.Linear(64, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, phone_vocab_size),
        )

    def encode(self, line_tokens, char_ids):
        tok_states, line_vecs, suffix, empty = self.line_enc(line_tokens)
        rhyme_vec = self.rhyme(suffix, empty)
        
        song_seq = self.song_enc(line_vecs, empty) # (B, L, D)
        char_seq = self.char_cnn(char_ids)        # (B, C, D)
        
        # Symmetric Co-Attention between song lines and character sequence
        song_fused, char_fused = self.co_attn(song_seq, char_seq)
        
        m_song = (~empty).unsqueeze(-1).float()
        song_vec = (song_fused * m_song).sum(1) / (m_song.sum(1) + 1e-9)
        
        m_char = (char_ids != 0).unsqueeze(-1).float()
        char_vec = (char_fused * m_char).sum(1) / (m_char.sum(1) + 1e-9)
        
        rep = self.repr_norm(torch.cat([song_vec, char_vec, rhyme_vec], dim=-1))
        return rep, tok_states

    def forward(self, line_tokens, char_ids, explicit_feats, adv_lambda: float = 1.0):
        rep, _ = self.encode(line_tokens, char_ids)
        tier = self.tier_head(torch.cat([rep, explicit_feats], dim=-1))
        elements = self.element_head(rep)
        source = self.source_head(grad_reverse(rep, adv_lambda))
        return tier, elements, source

    def pretrain_logits(self, line_tokens, char_ids):
        _, tok_states = self.encode(line_tokens, char_ids)
        return self.decoder(tok_states)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, T = 2, 12, 24
    model = BarsNet(phone_vocab_size=50, char_vocab_size=100)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    lines = torch.randint(N_SPECIALS, 50, (B, L, T))
    lines[:, 8:, :] = PAD
    lines[:, :, 20:] = PAD
    chars = torch.randint(1, 100, (B, 500))
    feats = torch.rand(B, NUM_ELEMENTS)

    tier, elements, source = model(lines, chars, feats)
    print("tier", tuple(tier.shape), "elements", tuple(elements.shape), "source", tuple(source.shape))
    loss = tier.sum() + elements.sum() + source.sum()
    loss.backward()
    print("backward ok; grad on line embed:", model.line_enc.embed.weight.grad is not None)

    logits = model.pretrain_logits(lines, chars)
    print("pretrain logits", tuple(logits.shape))
    logits.sum().backward()
    print("pretrain backward ok")
