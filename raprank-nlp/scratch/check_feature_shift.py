import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("C:/Users/Savyasachi Mishra/Desktop/Beatlyrix/raprank-nlp")
sys.path.insert(0, str(ROOT))

from services import bayesian_scoring_service as b

recs_train = b.load_train_records()
recs_eval = b.load_eval_records()

df_train = []
for r in recs_train:
    sc = b._axis_scores_from_lyrics(r["lyrics"])
    df_train.append({**sc, "tier": r["tier"], "src": "synthetic"})
df_train = pd.DataFrame(df_train)

df_eval = []
for r in recs_eval:
    sc = b._axis_scores_from_lyrics(r["lyrics"])
    df_eval.append({**sc, "tier": r["tier"], "src": "real"})
df_eval = pd.DataFrame(df_eval)

print("=== SYNTHETIC TRAINING POOL (Averages by Tier) ===")
print(df_train.groupby("tier")[["syllable", "vocabulary", "rhyme", "wordplay"]].mean())

print("\n=== REAL EVALUATION POOL (Averages by Tier) ===")
print(df_eval.groupby("tier")[["syllable", "vocabulary", "rhyme", "wordplay"]].mean())
