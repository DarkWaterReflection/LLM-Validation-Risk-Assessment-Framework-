"""RAGAS-backed faithfulness / hallucination evaluator (LLM-as-judge).

Implements the *same* ``Evaluator`` contract as the deterministic baseline, so
the risk engine, reporting, and swarm orchestration are untouched. RAGAS runs
in-process and makes its own judge-LLM calls (Anthropic or OpenAI via LangChain);
embeddings default to a local sentence-transformers model so no embeddings API key
is required.

All heavy imports are lazy and guarded — importing this module never pulls in
ragas/langchain, and a missing dependency or API key raises a clear, actionable
error rather than crashing the platform. Use :mod:`..factory` to select between
this and the deterministic evaluator at runtime.

Reproducibility note: LLM judges are only *approximately* deterministic even at
temperature 0. The deterministic evaluator is therefore retained as a QA cross-
check oracle (see ``factory.cross_check_faithfulness``).
"""

from __future__ import annotations

import math

from ...config import Settings
from ...logging_config import get_logger
from ...schemas import EvalRecord, MetricResult, ModuleResult, RiskDimension
from ..base import Evaluator
from ..faithfulness import FAITHFULNESS_FAIL

log = get_logger("ragas")


class RagasDependencyError(RuntimeError):
    """Raised when RAGAS or its provider wiring is unavailable."""


def _build_judge_llm(settings: Settings):
    """Construct a LangChain chat model wrapped for RAGAS."""
    from ragas.llms import LangchainLLMWrapper

    provider = settings.judge_provider.lower()
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RagasDependencyError(
                "langchain-anthropic not installed. Run: pip install '.[llm]'"
            ) from exc
        llm = ChatAnthropic(model=settings.judge_model, temperature=settings.judge_temperature)
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RagasDependencyError(
                "langchain-openai not installed. Run: pip install '.[llm]'"
            ) from exc
        llm = ChatOpenAI(model=settings.judge_model, temperature=settings.judge_temperature)
    else:  # pragma: no cover - validated upstream
        raise RagasDependencyError(f"Unknown judge_provider: {settings.judge_provider!r}")
    return LangchainLLMWrapper(llm)


def _build_embeddings(settings: Settings):
    """Local sentence-transformers embeddings wrapped for RAGAS (no API key)."""
    from ragas.embeddings import LangchainEmbeddingsWrapper

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:  # pragma: no cover - fall back to community package
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.judge_embedding_model)
    )


def _nan_safe_mean(values: list[float]) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return round(sum(clean) / len(clean), 4) if clean else 0.0


class RagasFaithfulnessEvaluator(Evaluator):
    """Faithfulness/hallucination via the RAGAS metric suite."""

    name = "faithfulness"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(self, records: list[EvalRecord]) -> ModuleResult:
        try:
            from ragas import EvaluationDataset, evaluate
            from ragas.metrics import (
                Faithfulness,
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                ResponseRelevancy,
            )
        except ImportError as exc:
            raise RagasDependencyError(
                "ragas not installed. Run: pip install '.[llm]'"
            ) from exc

        llm = _build_judge_llm(self.settings)
        embeddings = _build_embeddings(self.settings)

        samples = [
            {
                "user_input": r.question,
                "response": r.answer,
                "retrieved_contexts": r.contexts or [""],
                "reference": r.ground_truth or r.answer,
            }
            for r in records
        ]
        dataset = EvaluationDataset.from_list(samples)

        metrics = [
            Faithfulness(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            ResponseRelevancy(),
        ]
        log.info("Running RAGAS judge=%s on %d records", self.settings.judge_model, len(records))
        result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
        df = result.to_pandas()

        # Resolve column names by each metric's declared name (version-robust).
        faith_col = Faithfulness().name
        prec_col = LLMContextPrecisionWithReference().name
        rec_col = LLMContextRecall().name
        rel_col = ResponseRelevancy().name

        faith_vals = df[faith_col].tolist()
        failures: list[dict] = []
        for r, f in zip(records, faith_vals):
            if f is not None and not math.isnan(f) and f < FAITHFULNESS_FAIL:
                failures.append(
                    {
                        "id": r.id,
                        "faithfulness": round(float(f), 3),
                        "category": "unsupported_claim",
                        "explanation": (
                            f"RAGAS faithfulness {f:.2f} below {FAITHFULNESS_FAIL}: the "
                            "judge found claims not entailed by the retrieved context."
                        ),
                    }
                )

        n = max(len(records), 1)
        hallucination_rate = len(failures) / n

        return ModuleResult(
            module=self.name,
            dimension=RiskDimension.HALLUCINATION,
            n_records=len(records),
            metrics=[
                MetricResult(name="faithfulness", value=_nan_safe_mean(faith_vals),
                             explanation="RAGAS faithfulness (LLM-judged entailment)."),
                MetricResult(name="context_precision", value=_nan_safe_mean(df[prec_col].tolist()),
                             explanation="RAGAS context precision w/ reference."),
                MetricResult(name="context_recall", value=_nan_safe_mean(df[rec_col].tolist()),
                             explanation="RAGAS context recall."),
                MetricResult(name="answer_relevance", value=_nan_safe_mean(df[rel_col].tolist()),
                             explanation="RAGAS response relevancy (embedding-based)."),
                MetricResult(name="hallucination_rate", value=hallucination_rate,
                             higher_is_better=False,
                             explanation="Share of responses below the faithfulness threshold."),
            ],
            failures=failures,
            summary={
                "hallucination_rate": hallucination_rate,
                "n_failures": len(failures),
                "judge_model": 0.0,  # numeric map; model id recorded in metadata/logs
            },
        )
