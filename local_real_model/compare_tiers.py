"""
Compares the synthetic-trained tier classifier (raprank-nlp's own
services/rf_quality_service.py, trained on corpus/data/ + corpus/synthetic_data/)
against this folder's real-only baseline (train_real_baseline.py, trained
on corpus/data/ alone) for the same lyrics -- a quick sanity check for
whenever a live score looks off, without needing to re-derive the diagnosis
by hand each time.

Usage:
    python local_real_model/compare_tiers.py path/to/lyrics.txt
    python local_real_model/compare_tiers.py --corpus-track kr-na vyanjan
    python local_real_model/compare_tiers.py --lang en path/to/english_lyrics.txt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

NLP_ROOT = Path(__file__).resolve().parent.parent / "raprank-nlp"
sys.path.insert(0, str(NLP_ROOT))

from services.bayesian_scoring_service import _axis_scores_from_lyrics  # noqa: E402

REAL_MODEL_PATH = Path(__file__).resolve().parent / "real_only_rf_model.pkl"
REAL_MODEL_PATH_EN = Path(__file__).resolve().parent / "real_only_rf_model_en.pkl"
SYNTHETIC_MODEL_PATH = NLP_ROOT / "corpus" / "synthetic_data" / "_rf_model.pkl"


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(bundle: dict, lyrics: str) -> dict:
    scores = _axis_scores_from_lyrics(lyrics)
    X = [[scores[axis] for axis in bundle["feature_names"]]]
    probs = bundle["rf"].predict_proba(X)[0]
    classes = bundle["rf"].classes_
    top = int(probs.argmax())
    return {
        "tier": classes[top],
        "confidence": round(float(probs[top]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "axis_scores": scores,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lyrics_file", nargs="?", help="path to a UTF-8 text file of lyrics")
    ap.add_argument("--corpus-track", nargs=2, metavar=("ARTIST_DIR", "TITLE_SLUG"),
                     help="pull lyrics from raprank-nlp/corpus/data/<artist>/<title>.json instead")
    ap.add_argument("--lang", choices=["hi", "en"], default="hi",
                     help="hi (default) compares against real_only_rf_model.pkl "
                          "(Hindi/Hinglish); en compares against real_only_rf_model_en.pkl "
                          "(English, trained by train_real_baseline_english.py)")
    args = ap.parse_args()

    real_model_path = REAL_MODEL_PATH_EN if args.lang == "en" else REAL_MODEL_PATH

    if args.corpus_track:
        import json
        artist_dir, title_slug = args.corpus_track
        d = NLP_ROOT / "corpus" / "data" / artist_dir
        match = next((p for p in d.glob("*.json") if title_slug in p.stem.lower()), None)
        if not match:
            print(f"No track matching '{title_slug}' found under {d}")
            return 1
        lyrics = json.loads(match.read_text(encoding="utf-8"))["lyrics"]
    elif args.lyrics_file:
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")
    else:
        ap.print_help()
        return 1

    if not real_model_path.exists():
        trainer = "train_real_baseline_english.py" if args.lang == "en" else "train_real_baseline.py"
        print(f"Real-only model not found at {real_model_path} -- run {trainer} first.")
        return 1
    if not SYNTHETIC_MODEL_PATH.exists():
        print(f"Synthetic-trained model not found at {SYNTHETIC_MODEL_PATH} -- "
              f"run `python -m services.rf_quality_service --train` inside raprank-nlp first.")
        return 1

    real_bundle = load(real_model_path)
    synth_bundle = load(SYNTHETIC_MODEL_PATH)

    real_pred = predict(real_bundle, lyrics)
    synth_pred = predict(synth_bundle, lyrics)

    if args.lang == "en":
        print("NOTE: SYNTHETIC-TRAINED is rf_quality_service's live model, trained "
              "primarily on Hindi/Hinglish synthetic data -- off-domain for English "
              "lyrics until Part C's English synthetic generation contributes samples.\n")

    print(f"{'':20} {'REAL-ONLY':>15} {'SYNTHETIC-TRAINED':>20}")
    print(f"{'predicted tier':20} {real_pred['tier']:>15} {synth_pred['tier']:>20}")
    print(f"{'confidence':20} {real_pred['confidence']:>15.1%} {synth_pred['confidence']:>20.1%}")
    print()
    print("axis scores (shared feature set):")
    for axis in real_pred["axis_scores"]:
        print(f"  {axis:15} {real_pred['axis_scores'][axis]:6.2f}")

    if real_pred["tier"] != synth_pred["tier"]:
        print(f"\n*** DISAGREEMENT: real-only says '{real_pred['tier']}', "
              f"synthetic-trained says '{synth_pred['tier']}' -- worth a manual look. ***")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
