"""Module 5 — Performance & Reliability assessment.

Classification quality (accuracy/precision/recall/F1) plus operational telemetry
(latency, throughput, cost). Supports head-to-head comparison of candidate models
and RAG configurations.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..schemas import MetricResult, ModuleResult, RiskDimension


@dataclass
class PerfSample:
    predicted_label: int   # 1 = positive
    actual_label: int
    latency_ms: float
    cost_usd: float = 0.0


@dataclass
class PerfRun:
    model: str
    samples: list[PerfSample] = field(default_factory=list)


def _prf(samples: list[PerfSample]) -> tuple[float, float, float, float]:
    tp = sum(s.predicted_label == 1 and s.actual_label == 1 for s in samples)
    fp = sum(s.predicted_label == 1 and s.actual_label == 0 for s in samples)
    fn = sum(s.predicted_label == 0 and s.actual_label == 1 for s in samples)
    tn = sum(s.predicted_label == 0 and s.actual_label == 0 for s in samples)
    total = max(tp + fp + fn + tn, 1)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


class PerformanceEvaluator:
    name = "performance"

    def evaluate(self, run: PerfRun) -> ModuleResult:
        acc, prec, rec, f1 = _prf(run.samples)
        lat = [s.latency_ms for s in run.samples] or [0.0]
        p95 = sorted(lat)[int(0.95 * (len(lat) - 1))]
        mean_lat = statistics.mean(lat)
        throughput = 1000.0 / mean_lat if mean_lat else 0.0
        total_cost = sum(s.cost_usd for s in run.samples)

        return ModuleResult(
            module=self.name,
            dimension=RiskDimension.OPERATIONAL,
            n_records=len(run.samples),
            metrics=[
                MetricResult(name="accuracy", value=round(acc, 4)),
                MetricResult(name="precision", value=round(prec, 4)),
                MetricResult(name="recall", value=round(rec, 4)),
                MetricResult(name="f1", value=round(f1, 4)),
                MetricResult(name="latency_p95_ms", value=round(p95, 2),
                             higher_is_better=False),
                MetricResult(name="throughput_rps", value=round(throughput, 2)),
                MetricResult(name="cost_usd_total", value=round(total_cost, 4),
                             higher_is_better=False),
            ],
            summary={"model": 0.0, "f1": round(f1, 4), "latency_p95_ms": round(p95, 2)},
        )

    def compare(self, runs: list[PerfRun]) -> list[ModuleResult]:
        return [self.evaluate(r) for r in runs]
