"""
Local-only quality-tier classifier trained EXCLUSIVELY on real English rap
lyrics -- corpus/real_corpus/data/ (35 artists, tiers from real_artists.py's
REAL_ARTIST_TIERS, baked into each record at fetch time) plus the 11 Kaggle
rap-lyrics-for-nlp artists (kaggle_rap_data/, tiers from this folder's
kaggle_artist_tiers.py, produced by derive_english_tiers.py). No synthetic
data at all.

Mirrors train_real_baseline.py's structure exactly (same RandomForestClassifier
+ GridSearchCV pattern, same AXES/_axis_scores_from_lyrics feature set), but
for English instead of Hindi/Hinglish -- kept as a side-by-side reference so
predictions from the synthetic-trained heads can be sanity-checked against a
model built only on real English songs, once synthetic English generation
(Part C) starts contributing to the live training data.

Lives OUTSIDE raprank-nlp/ and is fully gitignored (see BeatLyrix/.gitignore)
-- both source corpora (corpus/real_corpus/data/ and kaggle_rap_data/) are
scraped, copyrighted material that must never enter the tracked repo, even
indirectly through a trained artifact.

Usage (run with raprank-nlp's venv active):
    python local_real_model/train_real_baseline_english.py
    python local_real_model/compare_tiers.py --lang en path/to/lyrics.txt
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

NLP_ROOT = Path(__file__).resolve().parent.parent / "raprank-nlp"
sys.path.insert(0, str(NLP_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.bayesian_scoring_service import (  # noqa: E402
    AXES,
    _axis_scores_from_lyrics,
    load_real_corpus_records,
)

from derive_english_tiers import clean_kaggle_lyrics, KAGGLE_CSV_PATH  # noqa: E402
from kaggle_artist_tiers import KAGGLE_ARTIST_TIERS  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "real_only_rf_model_en.pkl"
RANDOM_STATE = 0
MIN_SAMPLES_PER_CLASS_FOR_HOLDOUT = 8
CV_FOLDS = 5
PARAM_GRID = {
    "rf__n_estimators": [100, 300],
    "rf__max_depth": [None, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__class_weight": ["balanced", "balanced_subsample"],
}


def load_kaggle_records() -> list[dict]:
    import pandas as pd

    if not KAGGLE_CSV_PATH.exists():
        return []
    df = pd.read_csv(KAGGLE_CSV_PATH)
    records: list[dict] = []
    for artist, group in df.groupby("artist"):
        tier = KAGGLE_ARTIST_TIERS.get(str(artist))
        if not tier:
            continue
        for raw in group["artist_verses"].tolist():
            lyrics = clean_kaggle_lyrics(str(raw))
            if len(lyrics) >= 50:
                records.append({"artist": str(artist), "tier": tier, "lyrics": lyrics})
    return records


def load_records() -> list[dict]:
    return load_real_corpus_records() + load_kaggle_records()


def build_training_frame(records: list[dict]):
    import pandas as pd

    rows, labels = [], []
    for rec in records:
        scores = _axis_scores_from_lyrics(rec["lyrics"])
        rows.append({axis: scores[axis] for axis in AXES})
        labels.append(rec["tier"])
    return pd.DataFrame(rows, columns=list(AXES)), pd.Series(labels)


def train() -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    records = load_records()
    if not records:
        raise RuntimeError(
            "No real English corpus found -- run "
            "`python -m corpus.real_corpus.fetch_real_corpus` inside raprank-nlp "
            "and/or check kaggle_rap_data/extracted/lyrics_raw.csv exists."
        )

    X, y = build_training_frame(records)
    classes = sorted(y.unique())
    if len(classes) < 2:
        raise RuntimeError(f"Need at least 2 distinct tiers to train a forest, found: {classes}")

    counts = y.value_counts()
    can_holdout = counts.min() >= MIN_SAMPLES_PER_CLASS_FOR_HOLDOUT

    accuracy = accuracy_std = cm = best_params = None
    n_folds = 0
    pipe = Pipeline([("rf", RandomForestClassifier(random_state=RANDOM_STATE))])

    if can_holdout:
        n_folds = min(CV_FOLDS, counts.min())
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

        search = GridSearchCV(pipe, PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=-1)
        search.fit(X, y)
        best_params = search.best_params_
        accuracy = float(search.best_score_)
        accuracy_std = float(search.cv_results_["std_test_score"][search.best_index_])

        best_pipe = Pipeline([("rf", RandomForestClassifier(
            random_state=RANDOM_STATE,
            **{k.split("__")[1]: v for k, v in best_params.items()},
        ))])
        y_pred = cross_val_predict(best_pipe, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred, labels=classes).tolist()
    else:
        print(
            f"NOTE: smallest class has only {counts.min()} sample(s) "
            f"(need >= {MIN_SAMPLES_PER_CLASS_FOR_HOLDOUT} for an honest k-fold CV) "
            f"-- skipping the hyperparameter sweep, using RandomForestClassifier "
            f"defaults with no trustworthy accuracy number.",
            file=sys.stderr,
        )

    rf_kwargs = {k.split("__")[1]: v for k, v in best_params.items()} if best_params else {"class_weight": "balanced"}
    rf_final = RandomForestClassifier(random_state=RANDOM_STATE, **rf_kwargs)
    rf_final.fit(X, y)

    importances = dict(zip(AXES, rf_final.feature_importances_.tolist()))

    return {
        "rf": rf_final,
        "feature_names": list(AXES),
        "classes": list(rf_final.classes_),
        "held_out_accuracy": accuracy,
        "held_out_accuracy_std": accuracy_std,
        "best_params": best_params,
        "cv_folds": n_folds,
        "confusion_matrix": cm,
        "confusion_matrix_labels": classes,
        "feature_importances": importances,
        "n_records": len(records),
    }


def save(bundle: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)


def main() -> int:
    print("Training real-English-corpus-only tier classifier (real_corpus/data/ + kaggle_rap_data/, no synthetic data)...")
    bundle = train()
    save(bundle)
    print(f"Saved to {MODEL_PATH}  ({bundle['n_records']} real English songs, "
          f"tiers from real_artists.py + kaggle_artist_tiers.py)")

    if bundle["held_out_accuracy"] is None:
        print("\nNo held-out accuracy available -- too few samples in the smallest "
              "tier for a trustworthy CV split.")
    else:
        print(f"\nHeld-out accuracy ({bundle['cv_folds']}-fold CV): "
              f"{bundle['held_out_accuracy']:.1%} +/- {bundle['held_out_accuracy_std']:.1%}")
        print(f"Best hyperparameters: {bundle['best_params']}")
        chance = 1.0 / len(bundle["classes"])
        print(f"Chance baseline ({len(bundle['classes'])} classes): {chance:.1%}")
        print(f"\nConfusion matrix (out-of-fold, rows=actual, cols=predicted), "
              f"labels={bundle['confusion_matrix_labels']}:")
        for label, row in zip(bundle["confusion_matrix_labels"], bundle["confusion_matrix"]):
            print(f"  {label:12} {row}")
        print("\nFeature importances (which axes drive tier prediction on REAL English songs only):")
        for axis, imp in sorted(bundle["feature_importances"].items(), key=lambda kv: -kv[1]):
            print(f"  {axis:15} {imp:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
