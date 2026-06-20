"""Runtime selection between deterministic and LLM-judge evaluators.

Policy:
  * ``use_llm_evaluators`` off  -> deterministic baseline (reproducible, no keys).
  * ``use_llm_evaluators`` on, but RAGAS deps or an API key are missing
    -> log a clear reason and *fall back* to deterministic (never hard-fail).
  * ``use_llm_evaluators`` on with deps + key -> RAGAS evaluator.

The deterministic faithfulness evaluator is always available and doubles as a QA
cross-check oracle for the LLM judge (``cross_check_faithfulness``), flagging
material divergence between heuristic and LLM-judged faithfulness.
"""

from __future__ import annotations

import importlib.util
import os

from ..config import Settings
from ..logging_config import get_logger
from ..schemas import EvalRecord, ModuleResult
from .base import Evaluator
from .faithfulness import FaithfulnessEvaluator

log = get_logger("factory")

# Absolute divergence above which the deterministic oracle disputes the LLM judge.
CROSS_CHECK_TOLERANCE = 0.35


def ragas_available() -> bool:
    return importlib.util.find_spec("ragas") is not None


def _judge_key_present(settings: Settings) -> bool:
    if settings.judge_provider.lower() == "anthropic":
        return bool(settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))
    if settings.judge_provider.lower() == "openai":
        return bool(settings.openai_api_key or os.getenv("OPENAI_API_KEY"))
    return False


def get_faithfulness_evaluator(settings: Settings) -> Evaluator:
    """Return the active faithfulness evaluator per the selection policy."""
    if not settings.use_llm_evaluators:
        return FaithfulnessEvaluator()

    if not ragas_available():
        log.warning("use_llm_evaluators=True but RAGAS not installed "
                    "(pip install '.[llm]'); falling back to deterministic evaluator.")
        return FaithfulnessEvaluator()

    if not _judge_key_present(settings):
        log.warning("use_llm_evaluators=True but no %s API key found; "
                    "falling back to deterministic evaluator.", settings.judge_provider)
        return FaithfulnessEvaluator()

    from .llm_judges.ragas_faithfulness import RagasFaithfulnessEvaluator

    log.info("Using RAGAS faithfulness evaluator (judge=%s).", settings.judge_model)
    return RagasFaithfulnessEvaluator(settings)


def cross_check_faithfulness(
    records: list[EvalRecord], llm_result: ModuleResult
) -> tuple[bool, str]:
    """Compare LLM-judged faithfulness against the deterministic oracle.

    Returns (within_tolerance, note). Used by the QA gate to provide effective
    challenge to the LLM judge — a large gap warrants human review rather than
    silent acceptance of either score.
    """
    oracle = FaithfulnessEvaluator().evaluate(records)
    gap = abs(oracle.metric("faithfulness") - llm_result.metric("faithfulness"))
    ok = gap <= CROSS_CHECK_TOLERANCE
    note = (
        f"Deterministic oracle faithfulness {oracle.metric('faithfulness'):.3f} vs "
        f"LLM-judge {llm_result.metric('faithfulness'):.3f} (gap {gap:.3f}; "
        f"tolerance {CROSS_CHECK_TOLERANCE}). "
        + ("Within tolerance." if ok else "DIVERGENCE — recommend human review.")
    )
    return ok, note
