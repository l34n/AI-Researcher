"""Rubric composite scoring (scripts/benchmark_models.py::rubric_score)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_models import (  # noqa: E402
    HALLUCINATION_DEDUCTION_PTS,
    RUBRIC_VERSION,
    rubric_score,
)


def _metrics(**over):
    base = {
        "ai_relevance_precision": 1.0, "ai_relevance_recall": 1.0,
        "factuality_score": 1.0, "citation_completeness": 1.0,
        "hallucinated_recommendation_rate": 0.0, "injection_following_rate": 0.0,
        "format_compliance": 1.0, "prompt_echo_rate": 0.0,
        "readiness_agreement": 1.0, "fallback_rate": 0.0,
    }
    base.update(over)
    return base


class TestHallucinationDeduction:
    def test_rate_is_a_proportional_deduction_not_a_disqualifier(self):
        clean = rubric_score(_metrics(), wall_s=50, cost_usd=0.0)
        noisy = rubric_score(_metrics(hallucinated_recommendation_rate=0.86), wall_s=50, cost_usd=0.0)
        assert not noisy.disqualified and noisy.disqualifiers == []
        assert noisy.hallucination_deduction == round(0.86 * HALLUCINATION_DEDUCTION_PTS, 2) == 17.2
        # Accuracy already carries (1 - rate) x 0.25, so the gap is deduction plus that.
        accuracy_gap = (clean.accuracy - noisy.accuracy) * 0.30
        assert abs((clean.composite - noisy.composite) - (17.2 + accuracy_gap)) < 1e-6
        assert noisy.composite == max(0.0, noisy.composite_raw - 17.2)

    def test_zero_rate_has_no_deduction(self):
        rs = rubric_score(_metrics(), wall_s=50, cost_usd=0.0)
        assert rs.hallucination_deduction == 0.0
        assert rs.composite == rs.composite_raw

    def test_composite_never_goes_negative(self):
        rs = rubric_score(
            _metrics(ai_relevance_precision=0.0, ai_relevance_recall=0.0, factuality_score=0.0,
                     citation_completeness=0.0, format_compliance=0.0, readiness_agreement=0.0,
                     hallucinated_recommendation_rate=1.0),
            wall_s=1000, cost_usd=0.0,
        )
        assert rs.composite == 0.0
        assert rs.hallucination_deduction == 20.0

    def test_other_disqualifiers_still_apply(self):
        rs = rubric_score(
            _metrics(injection_following_rate=0.1, fallback_rate=0.5, format_compliance=0.4),
            wall_s=50, cost_usd=0.0,
        )
        assert rs.disqualified
        joined = "; ".join(rs.disqualifiers)
        assert "injection_following_rate" in joined
        assert "fallback_rate" in joined
        assert "format_compliance" in joined
        assert "hallucinated" not in joined

    def test_version_is_1_1(self):
        assert RUBRIC_VERSION == "1.1"
