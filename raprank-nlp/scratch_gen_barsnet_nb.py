"""Generate kaggle_barsnet/train_barsnet.ipynb with inlined BarsNet model."""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(ROOT)

src = open(os.path.join(ROOT, "services", "barsnet.py"), encoding="utf-8").read()
model_code = src.split('if __name__ == "__main__":')[0]

cells = []


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.splitlines(keepends=True)})


code("""# BarsNet training — stage 1: masked-span pretraining, stage 2: fine-tune
import os, json, math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

random.seed(42); np.random.seed(42); torch.manual_seed(42)
device = torch.device('cpu')
if torch.cuda.is_available():
    try:
        torch.zeros(1, device='cuda'); device = torch.device('cuda')
    except Exception as e:
        print('CUDA incompatible:', e)
print('device:', device)""")

code(model_code)

code("""# Load dataset + meta
data_path = meta_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f == 'barsnet_dataset.json': data_path = os.path.join(root, f)
        if f == 'barsnet_meta.json':    meta_path = os.path.join(root, f)
assert data_path and meta_path, 'barsnet dataset files not found'
records = json.load(open(data_path, encoding='utf-8'))
meta    = json.load(open(meta_path, encoding='utf-8'))
BARS_VOCAB = meta['bars_vocab']; CHAR2IDX = meta['char2idx']
MAXT, MAXL, MAXC = meta['max_line_tokens'], meta['max_lines'], meta['max_chars']
TIER2IDX = {'elite':0,'mid':1,'commercial':2}
print(len(records), 'records | vocab', len(BARS_VOCAB), '| chars', len(CHAR2IDX))""")

code("""class BarsDataset(Dataset):
    def __init__(self, recs):
        self.recs = recs
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        lines = torch.zeros(MAXL, MAXT, dtype=torch.long)
        for li, ids in enumerate(r['lines'][:MAXL]):
            t = torch.tensor(ids[:MAXT], dtype=torch.long)
            lines[li, :len(t)] = t
        chars = torch.zeros(MAXC, dtype=torch.long)
        c = torch.tensor(r['chars'][:MAXC], dtype=torch.long)
        chars[:len(c)] = c
        feats = torch.tensor(r['features'], dtype=torch.float)
        tier = TIER2IDX.get(r['tier'], -1)
        source = 0 if r['source'] == 'real' else 1
        return {'lines': lines, 'chars': chars, 'feats': feats,
                'tier': torch.tensor(tier), 'source': torch.tensor(source)}

pretrain_recs = [r for r in records if r['split'] != 'val']
train_recs    = [r for r in records if r['split'] == 'train' and r['tier']]
val_recs      = [r for r in records if r['split'] == 'val']
print(f'pretrain {len(pretrain_recs)} | finetune-train {len(train_recs)} | val {len(val_recs)}')

model = BarsNet(phone_vocab_size=len(BARS_VOCAB), char_vocab_size=len(CHAR2IDX)+1).to(device)
print('params:', sum(p.numel() for p in model.parameters() if p.requires_grad))""")

code("""# Stage 1: masked-span pretraining
PRE_EPOCHS, MASK_P, SUFFIX_P = 8, 0.15, 0.25
pre_loader = DataLoader(BarsDataset(pretrain_recs), batch_size=16, shuffle=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

def mask_batch(lines):
    tgt = lines.clone()
    masked = lines.clone()
    real = lines > 4
    span = (torch.rand_like(lines.float()) < MASK_P) & real
    B, L, T = lines.shape
    lens = (lines != 0).sum(-1)
    pick = (torch.rand(B, L, device=lines.device) < SUFFIX_P) & (lens > 8)
    idx = torch.arange(T, device=lines.device).view(1, 1, T)
    back = (lens.unsqueeze(-1) - 1 - idx)
    suffix = pick.unsqueeze(-1) & (back >= 0) & (back < 6) & real
    m = span | suffix
    masked[m] = 2
    return masked, tgt, m

pre_hist = []
for ep in range(PRE_EPOCHS):
    model.train(); tot, nb = 0.0, 0
    for batch in tqdm(pre_loader, desc=f'pretrain {ep+1}/{PRE_EPOCHS}', leave=False):
        lines = batch['lines'].to(device); chars = batch['chars'].to(device)
        masked, tgt, m = mask_batch(lines)
        logits = model.pretrain_logits(masked, chars)
        loss = F.cross_entropy(logits[m], tgt[m]) if m.any() else torch.tensor(0.0, device=device)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += loss.item(); nb += 1
    pre_hist.append(tot / max(nb, 1))
    print(f'pretrain ep {ep+1}: masked-CE {pre_hist[-1]:.4f}')
torch.save(model.state_dict(), 'barsnet_pretrained.pt')""")

code("""# Stage 2: fine-tune (balanced sampler + split-task + adversarial)
FT_EPOCHS, EL_LAMBDA, ADV_LAMBDA = 25, 0.3, 0.1
tier_trainable = [r['source'] in ('real', 'consented') for r in train_recs]

counts = {}
for r in train_recs: counts[r['tier']] = counts.get(r['tier'], 0) + 1
weights = [1.0 / counts[r['tier']] for r in train_recs]
sampler = WeightedRandomSampler(weights, num_samples=len(train_recs), replacement=True)

class FTDataset(BarsDataset):
    def __getitem__(self, i):
        d = super().__getitem__(i)
        d['tier_on'] = torch.tensor(1.0 if tier_trainable[i] else 0.0)
        return d

ft_loader  = DataLoader(FTDataset(train_recs), batch_size=16, sampler=sampler)
val_loader = DataLoader(BarsDataset(val_recs), batch_size=32, shuffle=False)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)

best_val, history = 0.0, []
for ep in range(FT_EPOCHS):
    model.train(); tl, correct, seen, nb = 0.0, 0, 0, 0
    for b in tqdm(ft_loader, desc=f'ft {ep+1}/{FT_EPOCHS}', leave=False):
        lines, chars = b['lines'].to(device), b['chars'].to(device)
        feats, tiers = b['feats'].to(device), b['tier'].to(device)
        src, ton = b['source'].to(device), b['tier_on'].to(device)
        tier_logits, elems, src_logits = model(lines, chars, feats, adv_lambda=ADV_LAMBDA)
        ce = F.cross_entropy(tier_logits, tiers, reduction='none')
        tier_loss = (ce * ton).sum() / ton.sum().clamp(min=1.0)
        el_loss   = F.mse_loss(elems, feats.clamp(0, 1))
        adv_loss  = F.cross_entropy(src_logits, src)
        loss = tier_loss + EL_LAMBDA * el_loss + adv_loss
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += tier_loss.item(); nb += 1
        keep = ton > 0
        correct += (tier_logits.argmax(-1)[keep] == tiers[keep]).sum().item()
        seen += int(keep.sum())
    sched.step()

    model.eval(); vc, vt = 0, 0
    with torch.no_grad():
        for b in val_loader:
            tier_logits, _, _ = model(b['lines'].to(device), b['chars'].to(device), b['feats'].to(device))
            vc += (tier_logits.argmax(-1).cpu() == b['tier']).sum().item()
            vt += len(b['tier'])
    val_acc = vc / vt
    history.append({'epoch': ep+1, 'tier_loss': tl/max(nb,1),
                    'train_acc': correct/max(seen,1), 'val_acc': val_acc})
    print(f"ft ep {ep+1:02d}: tier_loss={tl/max(nb,1):.4f} train_acc={correct/max(seen,1):.4f} val_acc={val_acc:.4f}")
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), 'barsnet.pt')
        print(f'  [*] new best {best_val:.4f} -> barsnet.pt')

print('best val acc:', best_val)
json.dump({'pretrain': pre_hist, 'finetune': history}, open('barsnet_history.json', 'w'), indent=2)
json.dump(meta, open('barsnet_meta_out.json', 'w'))""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 4}

for i, c in enumerate(nb["cells"]):
    compile("".join(c["source"]), f"<cell{i}>", "exec")

out_dir = os.path.join(BASE, "kaggle_barsnet")
os.makedirs(out_dir, exist_ok=True)
json.dump(nb, open(os.path.join(out_dir, "train_barsnet.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({
    "id": "mishmay/train-barsnet",
    "title": "Train BarsNet",
    "code_file": "train_barsnet.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": ["mishmay/dhh-lyrics-dataset"],
    "competition_sources": [], "kernel_sources": [],
    "docker_image_pinning_type": "latest",
}, open(os.path.join(out_dir, "kernel-metadata.json"), "w"), indent=2)
print("kaggle_barsnet/ prepared - all cells compile")
