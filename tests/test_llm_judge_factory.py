"""Tests for the deterministic/LLM-judge evaluator selection policy.

These tests stay green with NO RAGAS install and NO API key — they assert the
*fallback* and *cross-check* contracts. The live RAGAS path is exercised only
when both deps and a key are present (skipped otherwise).
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from llm_validation_platform import sample_data
from llm_validation_platform.config import Settings
from llm_validation_platform.evaluations.factory import (
    CROSS_CHECK_TOLERANCE,
    cross_check_faithfulness,
    get_faithfulness_evaluator,
    ragas_available,
)
from llm_validation_platform.evaluations.faithfulness import FaithfulnessEvaluator

_HAS_RAGAS = importlib.util.find_spec("ragas") is not None
_HAS_KEY = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))


def test_deterministic_when_llm_disabled():
    ev = get_faithfulness_evaluator(Settings(use_llm_evaluators=False))
    assert isinstance(ev, FaithfulnessEvaluator)


def test_falls_back_when_deps_or_key_missing():
    # use_llm_evaluators=True but (in CI) no ragas/key -> must NOT crash.
    ev = get_faithfulness_evaluator(Settings(use_llm_evaluators=True))
    if not (_HAS_RAGAS and _HAS_KEY):
        assert isinstance(ev, FaithfulnessEvaluator)


def test_cross_check_agrees_with_itself():
    records = sample_data.faithfulness_records()
    det_result = FaithfulnessEvaluator().evaluate(records)
    ok, note = cross_check_faithfulness(records, det_result)
    assert ok is True  # oracle vs itself => zero gap
    assert "within tolerance" in note.lower()


def test_cross_check_flags_divergence():
    records = sample_data.faithfulness_records()
    fake = FaithfulnessEvaluator().evaluate(records)
    # Force a divergent LLM-judge faithfulness value.
    for m in fake.metrics:
        if m.name == "faithfulness":
            m.value = min(1.0, m.value + CROSS_CHECK_TOLERANCE + 0.2)
    ok, note = cross_check_faithfulness(records, fake)
    assert ok is False
    assert "divergence" in note.lower()


def test_ragas_available_reflects_environment():
    assert ragas_available() == _HAS_RAGAS


@pytest.mark.skipif(not (_HAS_RAGAS and _HAS_KEY),
                    reason="RAGAS + API key required for live LLM-judge run")
def test_ragas_live_run_shape():  # pragma: no cover - integration only
    settings = Settings(use_llm_evaluators=True)
    ev = get_faithfulness_evaluator(settings)
    result = ev.evaluate(sample_data.faithfulness_records())
    assert {"faithfulness", "context_precision", "context_recall",
            "answer_relevance", "hallucination_rate"} <= {m.name for m in result.metrics}
