"""
SVM quality-tier classifier -- a supervised alternative to
services/bayesian_scoring_service.py, trained on the SAME AXES feature set
(imported verbatim, not re-derived, so the two heads are directly comparable
side by side) but on `sklearn.svm.SVC` instead of a naive-Bayes network.

Ground truth comes from corpus/synthetic_data/ -- every record written by
corpus/synthetic/generate.py, generate_with_reference.py, and (new)
generate_from_consented.py, plus corpus/real_corpus/ artist-tier records
(all reused via bayesian_scoring_service.load_all_records()). Records from
generate_from_consented.py trace back to real, CONSENTED seed lyrics (the
project owner's own writing + named collaborators who agreed to this use),
tier-labeled by the seed owner's own judgment -- the first genuinely human-
grounded label in this pipeline, not an assumed/derived one.

This is the first SUPERVISED head in the project: unlike percentiles (ECDF)
and GMM (unsupervised clustering), an SVM needs real labels to mean anything,
which is why it waited until labeled data existed at all.

This module intentionally does NOT replace main.py's linear total_score, nor
does it feed the live /analyze endpoint yet -- it stays an offline/CLI
comparison tool (like bayesian_scoring_service.py) until held-out accuracy is
judged trustworthy enough to wire in.

Usage:
    python -m services.svm_quality_service --train        # fit + save + report held-out accuracy
    python -m services.svm_quality_service --eval-corpus   # predicted tier for every training sample
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

MODEL_PATH = SYNTHETIC_DATA_DIR / "_svm_model.pkl"
RANDOM_STATE = 0
# Below this many samples PER CLASS, even a k-fold split can't form a fold per
# class (sklearn requires n_splits <= min class count), and any accuracy
# number from a near-empty fold would be noise, not signal. Skip the
# honest-report step below this and say so plainly, rather than crash or
# fake a number.
MIN_SAMPLES_PER_CLASS_FOR_HOLDOUT = 8
CV_FOLDS = 5
# Small grid over the two hyperparameters that matter most for an RBF SVC at
# this data size -- C (margin/regularization tradeoff) and gamma (kernel
# width). Defaults (C=1, gamma="scale") are rarely optimal; a cheap sweep
# over an order of magnitude each way costs little at ~a few hundred samples.
PARAM_GRID = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1, 1],
}


def build_training_frame(records: list[dict]):
    """(X, y) -- X is a DataFrame of raw AXES feature values, y is the tier label.
    Unlike bayesian_scoring_service's discretised bins (needed for its naive-Bayes
    CPTs), SVC works directly on continuous features -- no binning here."""
    import pandas as pd

    rows, labels = [], []
    for rec in records:
        scores = _axis_scores_from_lyrics(rec["lyrics"])
        rows.append({axis: scores[axis] for axis in AXES})
        labels.append(rec["tier"])
    return pd.DataFrame(rows, columns=list(AXES)), pd.Series(labels)


def train(records: list[dict] | None = None) -> dict:
    """Standardize features, hyperparameter-sweep an RBF-kernel SVC over a
    stratified k-fold cross-validation (not a single train/test split -- with
    only a few hundred samples and a skewed tier distribution, one split's
    accuracy has real variance; k-fold reports a mean +/- std instead, and
    GridSearchCV picks C/gamma rather than trusting sklearn's untuned
    defaults). This is the first held-out accuracy number anywhere in this
    pipeline, because it's the first genuinely supervised head.
    Returns a bundle {scaler, svm, feature_names, classes, held_out_accuracy,
    held_out_accuracy_std, best_params, confusion_matrix, ...}."""
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

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
        raise RuntimeError(f"Need at least 2 distinct tiers to train an SVM, found: {classes}")

    counts = y.value_counts()
    can_holdout = counts.min() >= MIN_SAMPLES_PER_CLASS_FOR_HOLDOUT

    accuracy = accuracy_std = None
    cm = None
    best_params = None
    n_folds = 0

    # Tiers won't be evenly represented (few "elite" consented seeds vs many
    # synthetic-from-scratch "commercial" samples) -- class_weight="balanced"
    # keeps the majority tier from dominating the decision boundary, in both
    # the CV search below and the final refit.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    if can_holdout:
        n_folds = min(CV_FOLDS, counts.min())
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

        search = GridSearchCV(pipe, PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=-1)
        search.fit(X, y)
        best_params = search.best_params_
        accuracy = float(search.best_score_)
        accuracy_std = float(search.cv_results_["std_test_score"][search.best_index_])

        # Confusion matrix from out-of-fold predictions at the best
        # hyperparameters -- every prediction comes from a fold that never
        # trained on that row, so this is still an honest held-out view.
        best_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced",
                        random_state=RANDOM_STATE, **{k.split("__")[1]: v for k, v in best_params.items()})),
        ])
        y_pred = cross_val_predict(best_pipe, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred, labels=classes).tolist()
    else:
        print(
            f"NOTE: smallest class has only {counts.min()} sample(s) "
            f"(need >= {MIN_SAMPLES_PER_CLASS_FOR_HOLDOUT} for an honest k-fold CV) "
            f"-- skipping the hyperparameter sweep and accuracy report, using SVC "
            f"defaults. Training on all data with NO trustworthy accuracy number yet; "
            f"generate more samples per tier before believing any prediction from this model.",
            file=sys.stderr,
        )

    # Refit on ALL data for the saved/served model, using the CV-selected
    # hyperparameters when available -- the CV above (when it happens) is
    # purely for the honest accuracy report and hyperparameter choice, the
    # deployed model itself always trains on every sample.
    svm_kwargs = {k.split("__")[1]: v for k, v in best_params.items()} if best_params else {}
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X)
    svm_final = SVC(kernel="rbf", probability=True, class_weight="balanced",
                     random_state=RANDOM_STATE, **svm_kwargs)
    svm_final.fit(X_all_s, y)

    return {
        "scaler": scaler,
        "svm": svm_final,
        "feature_names": list(AXES),
        "classes": list(svm_final.classes_),
        "held_out_accuracy": accuracy,          # None if too little data per class
        "held_out_accuracy_std": accuracy_std,  # None if too little data per class
        "best_params": best_params,             # None if too little data per class
        "cv_folds": n_folds,
        "confusion_matrix": cm,                 # None if too little data per class
        "confusion_matrix_labels": classes,
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
    X_s = bundle["scaler"].transform(X)
    probs = bundle["svm"].predict_proba(X_s)[0]
    classes = bundle["svm"].classes_
    top = int(probs.argmax())
    return {
        "tier": classes[top],
        "confidence": round(float(probs[top]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train / evaluate the SVM quality-tier classifier")
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
                  "smallest tier for a trustworthy test split. Generate more samples "
                  "per tier (see the NOTE above) before treating this model's "
                  "predictions as meaningful.")
        else:
            print(f"\nHeld-out accuracy ({bundle['cv_folds']}-fold CV): "
                  f"{bundle['held_out_accuracy']:.1%} +/- {bundle['held_out_accuracy_std']:.1%}")
            print(f"Best hyperparameters: {bundle['best_params']}")
            chance = 1.0 / len(bundle["classes"])
            print(f"Chance baseline ({len(bundle['classes'])} classes): {chance:.1%}")
            print(f"\nConfusion matrix (out-of-fold, rows=actual, cols=predicted), labels={bundle['confusion_matrix_labels']}:")
            for label, row in zip(bundle["confusion_matrix_labels"], bundle["confusion_matrix"]):
                print(f"  {label:12} {row}")
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
