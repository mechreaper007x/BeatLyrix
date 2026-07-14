"""
Corpus-level calibration checks.

These assert *distributional* health of the local scorers over the real corpus
rather than brittle per-song values: scores must be discriminating (non-zero
spread), must not saturate at 100 (a classic small-sample / low-threshold bug),
and elite technical lyricists should, on average, out-score the more commercial
end of the roster on the rhyme axis.

Skips if the corpus is missing or too small to be statistically meaningful.
"""
import statistics as stats

import pytest

from services import (
    alliteration_service, rhyme_service, syllable_service,
    vocabulary_service, wordplay_service,
)
from services.lyrical_compiler import compile_lyrics


def _summary(tracks):
    rows = []
    for t in tracks:
        ly = t["lyrics"]
        rows.append({
            "artist": t["artist"],
            "syllable": syllable_service.calculate(ly)[0],
            "rhyme": rhyme_service.calculate(ly)[0],
            "alliteration": alliteration_service.calculate(ly)[0],
            "vocabulary": vocabulary_service.calculate(ly)[0],
            "wordplay": wordplay_service.calculate(ly)[0],
            # live /analyze endpoint's rhyme_score -- a separate formula from
            # rhyme_service (dead for live scoring, see "rhyme" above).
            "llpc_rhyme": compile_lyrics(ly)["rhyme_complexity"],
        })
    return rows


@pytest.fixture(scope="module")
def summary(corpus_tracks):
    if len(corpus_tracks) < 50:
        pytest.skip("corpus too small for calibration")
    return _summary(corpus_tracks)


METRICS = ["syllable", "rhyme", "alliteration", "vocabulary", "wordplay", "llpc_rhyme"]


class TestDistribution:
    @pytest.mark.parametrize("metric", METRICS)
    def test_has_spread(self, summary, metric):
        """A metric that returns the same value for every song discriminates nothing."""
        vals = [r[metric] for r in summary]
        assert stats.pstdev(vals) > 3.0, f"{metric} has near-zero spread"

    @pytest.mark.parametrize("metric", METRICS)
    def test_mean_in_sane_band(self, summary, metric):
        vals = [r[metric] for r in summary]
        mean = stats.mean(vals)
        assert 5.0 < mean < 96.0, f"{metric} mean={mean:.1f} is stuck/saturated"

    @pytest.mark.parametrize("metric", METRICS)
    def test_not_saturated_at_100(self, summary, metric):
        """No metric should peg a large fraction of real songs at a perfect 100."""
        vals = [r[metric] for r in summary]
        frac_100 = sum(1 for v in vals if v >= 99.5) / len(vals)
        assert frac_100 < 0.25, f"{metric}: {frac_100:.0%} of songs score ~100"


class TestSourceDiscrimination:
    """The scorers must distinguish between sources, not flatten everyone to the
    same score. Groups are read from whatever `artist` labels the corpus carries
    — no labels are hard-coded here."""

    def _group_means(self, summary, metric, min_tracks=5):
        from collections import defaultdict
        groups = defaultdict(list)
        for r in summary:
            groups[r["artist"]].append(r[metric])
        return [stats.mean(v) for v in groups.values() if len(v) >= min_tracks]

    @pytest.mark.parametrize("metric", ["rhyme", "wordplay", "vocabulary", "llpc_rhyme"])
    def test_metric_separates_sources(self, summary, metric):
        means = self._group_means(summary, metric)
        if len(means) < 3:
            pytest.skip("need >= 3 sources with >= 5 tracks each")
        # A metric that assigns near-identical means to every source discriminates
        # nothing; a real one spreads technical vs commercial material apart.
        assert max(means) - min(means) > 5.0

