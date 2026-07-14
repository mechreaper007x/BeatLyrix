"""
Random Forest quality-tier classifier -- a second supervised alternative to
services/bayesian_scoring_service.py and services/svm_quality_service.py,
trained on the SAME AXES feature set (imported verbatim, so all three heads
are directly comparable side by side) but on
`sklearn.ensemble.RandomForestClassifier`.

Why a third head at this data size (~a few hundred samples, 10 axes):
unlike the RBF SVM, a forest needs no feature scaling, splits are made
per-feature (more robust to a skewed tier distribution -- e.g. 2 elite / 4
mid / 21 commercial consented seeds -- than a global margin), and it exposes
feature importances directly, which neither the SVM nor the Bayesian net's
CPTs give you: which of the 10 axes actually drive tier prediction.

Ground truth comes from corpus/synthetic_data/ (see svm_quality_service.py's
docstring for the full provenance chain back to consented seeds).

Not wired into the live /analyze endpoint -- stays an offline/CLI comparison
tool, like the other two heads, until one's accuracy earns that.

Usage:
    python -m services.rf_quality_service --train        # fit + save + report held-out accuracy
    python -m services.rf_quality_service --eval-corpus   # predicted tier for every training sample
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.bayesian_scoring_service import (  # noqa: E402
    AXES,
    SYNTHETIC_DATA_DIR,
    _axis_scores_from_lyrics,
    load_all_records,
)
from services.svm_quality_service import build_training_frame  # noqa: E402

MODEL_PATH = SYNTHETIC_DATA_DIR / "_rf_model.pkl"
RANDOM_STATE = 0
MIN_SAMPLES_PER_CLASS_FOR_HOLDOUT = 8
CV_FOLDS = 5
# Small grid over the hyperparameters that matter most for a forest at this
# data size -- n_estimators (variance reduction), max_depth (overfit control
# on a few hundred samples), min_samples_leaf (further overfit control). A
# wider sweep additionally tried min_samples_split and max_features -- both
# moved held-out accuracy by <0.5% (within CV noise) at several times the
# training cost, so they were dropped. class_weight is kept in the sweep,
# though: dropping it collapsed mid-tier recall (2/64 vs 11-15/64), so
# "balanced_subsample" vs "balanced" is doing real work for that thin class,
# not just noise -- worth the 2x combos.
PARAM_GRID = {
    "rf__n_estimators": [100, 300],
    "rf__max_depth": [None, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__class_weight": ["balanced", "balanced_subsample"],
}


def train(records: list[dict] | None = None) -> dict:
    """Hyperparameter-sweep a RandomForestClassifier over stratified k-fold CV
    (same rationale as svm_quality_service.train: one train/test split has
    too much variance at this sample size). No feature scaling needed --
    trees split per-feature, unaffected by differing axis scales.
    Returns a bundle {rf, feature_names, classes, held_out_accuracy,
    held_out_accuracy_std, best_params, confusion_matrix, feature_importances, ...}."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    if records is None:
        records = load_all_records()
    if not records:
        raise RuntimeError(
            "No training data found -- run corpus.synthetic.generate, "
            "generate_with_reference, and/or generate_from_consented first."
        )

    X, y = build_training_frame(records)
    classes = sorted(y.unique())
    if len(classes) < 2:
        raise RuntimeError(f"Need at least 2 distinct tiers to train a forest, found: {classes}")

    counts = y.value_counts()
    can_holdout = counts.min() >= MIN_SAMPLES_PER_CLASS_FOR_HOLDOUT

    accuracy = accuracy_std = None
    cm = None
    best_params = None
    n_folds = 0

    # class_weight defaults to "balanced" so the majority tier (commercial, from
    # the user's own larger seed pool) doesn't dominate the splits -- but it's
    # now also swept in PARAM_GRID against "balanced_subsample", so don't pass
    # it again explicitly here (that would collide with the grid's override).
    pipe = Pipeline([
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])

    if can_holdout:
        n_folds = min(CV_FOLDS, counts.min())
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

        search = GridSearchCV(pipe, PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=-1)
        search.fit(X, y)
        best_params = search.best_params_
        accuracy = float(search.best_score_)
        accuracy_std = float(search.cv_results_["std_test_score"][search.best_index_])

        best_pipe = Pipeline([
            ("rf", RandomForestClassifier(
                random_state=RANDOM_STATE,
                **{k.split("__")[1]: v for k, v in best_params.items()},
            )),
        ])
        y_pred = cross_val_predict(best_pipe, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred, labels=classes).tolist()
    else:
        print(
            f"NOTE: smallest class has only {counts.min()} sample(s) "
            f"(need >= {MIN_SAMPLES_PER_CLASS_FOR_HOLDOUT} for an honest k-fold CV) "
            f"-- skipping the hyperparameter sweep and accuracy report, using "
            f"RandomForestClassifier defaults. Training on all data with NO "
            f"trustworthy accuracy number yet.",
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
    }


def save(bundle: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load(path: Path = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_tier(bundle: dict, lyrics: str) -> dict:
    """{tier, confidence, probabilities} for a single verse."""
    scores = _axis_scores_from_lyrics(lyrics)
    X = [[scores[axis] for axis in bundle["feature_names"]]]
    probs = bundle["rf"].predict_proba(X)[0]
    classes = bundle["rf"].classes_
    top = int(probs.argmax())
    return {
        "tier": classes[top],
        "confidence": round(float(probs[top]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train / evaluate the Random Forest quality-tier classifier")
    ap.add_argument("--train", action="store_true", help="fit from corpus + synthetic data, save, report accuracy")
    ap.add_argument("--eval-corpus", action="store_true", help="print predicted tier for every training sample")
    args = ap.parse_args()

    if args.train:
        records = load_all_records()
        print(f"Training on {len(records)} samples...")
        bundle = train(records)
        save(bundle)
        print(f"Saved model to {MODEL_PATH}")
        if bundle["held_out_accuracy"] is None:
            print("\nNo held-out accuracy available yet -- too few samples in the "
                  "smallest tier for a trustworthy CV split. Generate more samples "
                  "per tier before treating this model's predictions as meaningful.")
        else:
            print(f"\nHeld-out accuracy ({bundle['cv_folds']}-fold CV): "
                  f"{bundle['held_out_accuracy']:.1%} +/- {bundle['held_out_accuracy_std']:.1%}")
            print(f"Best hyperparameters: {bundle['best_params']}")
            chance = 1.0 / len(bundle["classes"])
            print(f"Chance baseline ({len(bundle['classes'])} classes): {chance:.1%}")
            print(f"\nConfusion matrix (out-of-fold, rows=actual, cols=predicted), labels={bundle['confusion_matrix_labels']}:")
            for label, row in zip(bundle["confusion_matrix_labels"], bundle["confusion_matrix"]):
                print(f"  {label:12} {row}")
            print("\nFeature importances (which axes drive tier prediction):")
            for axis, imp in sorted(bundle["feature_importances"].items(), key=lambda kv: -kv[1]):
                print(f"  {axis:15} {imp:.3f}")
        return 0

    if args.eval_corpus:
        bundle = load()
        records = load_all_records()
        correct = 0
        for rec in records:
            pred = predict_tier(bundle, rec["lyrics"])
            correct += pred["tier"] == rec["tier"]
            print(f"  {rec['tier']:10} -> predicted={pred['tier']:10} conf={pred['confidence']:.2f}")
        if records:
            print(f"\nAccuracy vs. training label: {correct}/{len(records)} = {correct/len(records):.0%}")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
