"""Module 1 — Faithfulness / Hallucination evaluation.

Deterministic baseline implements the *shape* of the RAGAS metric suite
(faithfulness, context precision, context recall, answer relevance) using
transparent lexical-overlap proxies. Swap in RAGAS via ``use_llm_evaluators``.
"""

from __future__ import annotations

import statistics

from ..schemas import EvalRecord, MetricResult, ModuleResult, RiskDimension
from .base import Evaluator, context_support, cosine_bow

FAITHFULNESS_FAIL = 0.60  # support below this flags a likely hallucination


class FaithfulnessEvaluator(Evaluator):
    name = "faithfulness"

    def evaluate(self, records: list[EvalRecord]) -> ModuleResult:
        faith, ctx_prec, ctx_rec, ans_rel = [], [], [], []
        failures: list[dict] = []

        for r in records:
            f = context_support(r.answer, r.contexts)
            # context precision: of retrieved contexts, how many are on-topic vs answer
            prec = (
                statistics.mean(cosine_bow(c, r.answer) for c in r.contexts)
                if r.contexts
                else 0.0
            )
            # context recall: does context cover the ground truth (when available)
            rec = (
                context_support(r.ground_truth, r.contexts)
                if r.ground_truth
                else f
            )
            rel = cosine_bow(r.answer, r.question)

            faith.append(f)
            ctx_prec.append(prec)
            ctx_rec.append(rec)
            ans_rel.append(rel)

            if f < FAITHFULNESS_FAIL:
                failures.append(
                    {
                        "id": r.id,
                        "faithfulness": round(f, 3),
                        "category": "unsupported_claim",
                        "explanation": (
                            f"Only {f:.0%} of answer tokens are grounded in retrieved "
                            "context; remaining content is unsupported (hallucination risk)."
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
                MetricResult(name="faithfulness", value=_mean(faith),
                             explanation="Mean answer grounding in retrieved context."),
                MetricResult(name="context_precision", value=_mean(ctx_prec),
                             explanation="Topical alignment of retrieved contexts."),
                MetricResult(name="context_recall", value=_mean(ctx_rec),
                             explanation="Coverage of ground truth by retrieved context."),
                MetricResult(name="answer_relevance", value=_mean(ans_rel),
                             explanation="Alignment of answer to the question."),
                MetricResult(name="hallucination_rate", value=hallucination_rate,
                             higher_is_better=False,
                             explanation="Share of responses below faithfulness threshold."),
            ],
            failures=failures,
            summary={"hallucination_rate": hallucination_rate, "n_failures": len(failures)},
        )


def _mean(xs: list[float]) -> float:
    return round(statistics.mean(xs), 4) if xs else 0.0
