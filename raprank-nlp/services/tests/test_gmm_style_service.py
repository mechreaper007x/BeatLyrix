"""
GMM style-clustering service -- focus on the runtime contract & graceful
degradation. Avoids the heavy signature() call by monkeypatching the technical
feature extractor, and builds a tiny real GMM bundle so predict_proba is
exercised without needing the corpus or the Space.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from services import gmm_style_service as gss


def _tiny_bundle():
    """A real StandardScaler + GaussianMixture fit on tiny random 26-dim data."""
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(0)
    X = rng.rand(30, len(gss.FEATURE_NAMES)) * 100.0
    scaler = StandardScaler().fit(X)
    gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=0).fit(scaler.transform(X))
    return {"scaler": scaler, "gmm": gmm, "feature_names": list(gss.FEATURE_NAMES), "k": 3}


_GOOD_SEM = {"coherence_cosine": 0.4, "theme_cosine": 0.6, "pairwise_spread": 0.5, "mean_surprisal_nats": 6.0}


def test_score_style_returns_membership(monkeypatch):
    monkeypatch.setattr(gss, "_technical_features", lambda lyrics: {a: 50.0 for a in gss.TECHNICAL_AXES})
    out = gss.score_style("some bars\nmore bars", semantic_metrics=_GOOD_SEM, bundle=_tiny_bundle())
    assert out is not None
    assert set(out) == {"cluster", "confidence", "membership"}
    assert 0.0 <= out["confidence"] <= 1.0
    assert abs(sum(out["membership"].values()) - 1.0) < 1e-6   # soft split sums to 1
    assert out["cluster"] in out["membership"]


def test_missing_semantic_keys_degrade_to_none(monkeypatch):
    monkeypatch.setattr(gss, "_technical_features", lambda lyrics: {a: 50.0 for a in gss.TECHNICAL_AXES})
    # No live fetch should be attempted when metrics are present-but-incomplete.
    monkeypatch.setattr(gss, "_fetch_semantic_live", lambda lyrics: None)
    out = gss.score_style("bars", semantic_metrics={"coherence_cosine": 0.4}, bundle=_tiny_bundle())
    assert out is None


def test_no_model_degrades_to_none(monkeypatch):
    # With no bundle passed, score_style calls load(); if the model is absent
    # that raises FileNotFoundError -> caught -> None.
    def _boom(*a, **k):
        raise FileNotFoundError("no model")
    monkeypatch.setattr(gss, "load", _boom)
    assert gss.score_style("bars", semantic_metrics=_GOOD_SEM, bundle=None) is None


def test_cache_id_from_path():
    assert gss._cache_id_from_path("corpus/data/kr-na/no-cap.json") == "kr-na/no-cap"
    assert gss._cache_id_from_path(r"corpus\data\emiway-bantai\khatam.json") == "emiway-bantai/khatam"
