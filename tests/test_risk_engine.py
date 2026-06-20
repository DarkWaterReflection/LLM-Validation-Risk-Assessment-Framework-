"""Risk-scoring validation tests — the audit-critical logic."""

from __future__ import annotations

import pytest

from llm_validation_platform.risk_scoring.engine import RiskEngine
from llm_validation_platform.risk_scoring.framework import (
    DIMENSION_WEIGHTS,
    score_to_level,
    validate_weights,
)
from llm_validation_platform.schemas import (
    DimensionRisk,
    MetricResult,
    ModuleResult,
    RiskDimension,
    RiskLevel,
)


def test_weights_sum_to_one():
    validate_weights()
    assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9


@pytest.mark.parametrize(
    "score,expected",
    [(0, RiskLevel.LOW), (24.9, RiskLevel.LOW), (25, RiskLevel.MODERATE),
     (49, RiskLevel.MODERATE), (50, RiskLevel.HIGH), (74, RiskLevel.HIGH),
     (75, RiskLevel.CRITICAL), (100, RiskLevel.CRITICAL)],
)
def test_band_boundaries(score, expected):
    assert score_to_level(score) == expected


def test_perfect_faithfulness_is_low_risk():
    m = ModuleResult(
        module="faithfulness", dimension=RiskDimension.HALLUCINATION, n_records=10,
        metrics=[MetricResult(name="faithfulness", value=1.0),
                 MetricResult(name="hallucination_rate", value=0.0,
                              higher_is_better=False)],
    )
    risk = RiskEngine()._hallucination(m)
    assert risk.score == 0.0
    assert risk.level == RiskLevel.LOW


def test_full_hallucination_is_critical():
    m = ModuleResult(
        module="faithfulness", dimension=RiskDimension.HALLUCINATION, n_records=10,
        metrics=[MetricResult(name="faithfulness", value=0.0),
                 MetricResult(name="hallucination_rate", value=1.0,
                              higher_is_better=False)],
    )
    risk = RiskEngine()._hallucination(m)
    assert risk.score == 100.0
    assert risk.level == RiskLevel.CRITICAL


def test_data_leakage_escalates_security():
    m = ModuleResult(
        module="safety", dimension=RiskDimension.SECURITY, n_records=4,
        metrics=[MetricResult(name="safety_violation_rate", value=0.25,
                              higher_is_better=False),
                 MetricResult(name="data_leakage_count", value=1.0,
                              higher_is_better=False)],
    )
    risk = RiskEngine()._security(m)
    assert risk.score == pytest.approx(65.0)  # 25 + 40 escalator


def test_critical_override_floors_composite_at_high():
    engine = RiskEngine()
    risks = [
        DimensionRisk(dimension=RiskDimension.HALLUCINATION, score=90,
                      level=RiskLevel.CRITICAL, rationale=""),
        DimensionRisk(dimension=RiskDimension.OPERATIONAL, score=0,
                      level=RiskLevel.LOW, rationale=""),
    ]
    score, level = engine.composite(risks)
    assert score >= 75.0
    assert level == RiskLevel.CRITICAL


def test_composite_is_deterministic():
    engine = RiskEngine()
    risks = [
        DimensionRisk(dimension=d, score=30.0, level=RiskLevel.MODERATE, rationale="")
        for d in RiskDimension
    ]
    a = engine.composite(risks)
    b = engine.composite(risks)
    assert a == b == (30.0, RiskLevel.MODERATE)
