"""Module 3 — Robustness & Stability under input perturbation.

A robust model returns semantically consistent answers when the prompt is
perturbed (paraphrase, typos, reordering, synonym swaps) without changing intent.
Consistency is measured as similarity between the baseline answer and each
perturbed answer.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..schemas import MetricResult, ModuleResult, RiskDimension
from .base import cosine_bow

CONSISTENCY_FAIL = 0.55  # answers below this similarity are considered unstable


@dataclass
class RobustnessProbe:
    probe_id: str
    baseline_answer: str
    perturbed_answers: dict[str, str] = field(default_factory=dict)  # type -> answer


class RobustnessEvaluator:
    name = "robustness"

    def evaluate(self, probes: list[RobustnessProbe]) -> ModuleResult:
        per_type: dict[str, list[float]] = {}
        all_sims: list[float] = []
        failures: list[dict] = []

        for probe in probes:
            for ptype, ans in probe.perturbed_answers.items():
                sim = cosine_bow(probe.baseline_answer, ans)
                per_type.setdefault(ptype, []).append(sim)
                all_sims.append(sim)
                if sim < CONSISTENCY_FAIL:
                    failures.append(
                        {
                            "probe_id": probe.probe_id,
                            "perturbation": ptype,
                            "similarity": round(sim, 3),
                            "category": "instability",
                            "explanation": (
                                f"'{ptype}' perturbation changed the answer "
                                f"(similarity {sim:.2f} < {CONSISTENCY_FAIL})."
                            ),
                        }
                    )

        n = max(len(all_sims), 1)
        stability = statistics.mean(all_sims) if all_sims else 1.0
        unstable_rate = len(failures) / n
        # Robustness score blends mean consistency with the failure rate.
        robustness = round(0.7 * stability + 0.3 * (1 - unstable_rate), 4)

        metrics = [
            MetricResult(name="semantic_consistency", value=round(stability, 4),
                         explanation="Mean answer similarity across all perturbations."),
            MetricResult(name="robustness_score", value=robustness,
                         explanation="Blended stability + pass-rate score."),
            MetricResult(name="instability_rate", value=round(unstable_rate, 4),
                         higher_is_better=False,
                         explanation="Share of perturbations causing answer drift."),
        ]
        summary = {f"consistency::{t}": round(statistics.mean(v), 4) for t, v in per_type.items()}
        summary["instability_rate"] = unstable_rate

        return ModuleResult(
            module=self.name,
            dimension=RiskDimension.ROBUSTNESS,
            n_records=len(probes),
            metrics=metrics,
            failures=failures,
            summary=summary,
        )
