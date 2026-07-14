"""
Local-only quality-tier classifier trained EXCLUSIVELY on the real, scraped
Hindi/Hinglish corpus (raprank-nlp/corpus/data/, artist-level tier labels
from raprank-nlp/corpus/real_corpus/indian_tiers.py) -- no synthetic data at
all. Kept as a side-by-side reference so predictions from the
synthetic-trained heads (raprank-nlp/services/rf_quality_service.py,
svm_quality_service.py, bayesian_scoring_service.py) can be checked against
what a model built only on real songs would say, whenever a score looks
suspicious (like the KR$NA rhyme-complexity bug this was written to guard
against).

Lives OUTSIDE raprank-nlp/ and is fully gitignored (see BeatLyrix/.gitignore)
-- the corpus/data/ lyric text this is derived from is scraped, copyrighted
material that must never enter the tracked repo, even indirectly through a
trained artifact.

Usage (run with raprank-nlp's venv active):
    python local_real_model/train_real_baseline.py
    python local_real_model/compare_tiers.py path/to/lyrics.txt
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

NLP_ROOT = Path(__file__).resolve().parent.parent / "raprank-nlp"
sys.path.insert(0, str(NLP_ROOT))

from services.bayesian_scoring_service import (  # noqa: E402
    AXES,
    _axis_scores_from_lyrics,
    load_indian_corpus_records,
)

MODEL_PATH = Path(__file__).resolve().parent / "real_only_rf_model.pkl"
RANDOM_STATE = 0
MIN_SAMPLES_PER_CLASS_FOR_HOLDOUT = 8
CV_FOLDS = 5
PARAM_GRID = {
    "rf__n_estimators": [100, 300],
    "rf__max_depth": [None, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__class_weight": ["balanced", "balanced_subsample"],
}


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

    records = load_indian_corpus_records()
    if not records:
        raise RuntimeError(
            "No real corpus found -- run `python -m corpus.scrape_corpus` "
            "inside raprank-nlp first."
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
    print("Training real-corpus-only tier classifier (no synthetic data)...")
    bundle = train()
    save(bundle)
    print(f"Saved to {MODEL_PATH}  ({bundle['n_records']} real songs, tiers from indian_tiers.py)")

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
        print("\nFeature importances (which axes drive tier prediction on REAL songs only):")
        for axis, imp in sorted(bundle["feature_importances"].items(), key=lambda kv: -kv[1]):
            print(f"  {axis:15} {imp:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
