"""End-to-end validation pipeline.

This is the single in-process orchestration the Ruflo agent swarm coordinates:
each agent owns one stage, but the deterministic pipeline below is what every
agent ultimately calls so results are reproducible whether run by one process or
ten agents.
"""

from __future__ import annotations

from ..config import Settings
from ..evaluations.bias import BiasEvaluator, MatchedPair
from ..evaluations.factory import get_faithfulness_evaluator
from ..evaluations.performance import PerformanceEvaluator, PerfRun
from ..evaluations.robustness import RobustnessEvaluator, RobustnessProbe
from ..evaluations.safety import SafetyEvaluator, SafetyProbe
from ..governance.model_card import completeness_score
from ..logging_config import get_logger
from ..risk_scoring.engine import RiskEngine
from ..schemas import EvalRecord, ModuleResult, ValidationResult

log = get_logger("pipeline")


class ValidationPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = RiskEngine()

    def run(
        self,
        *,
        faithfulness_records: list[EvalRecord],
        matched_pairs: list[MatchedPair],
        robustness_probes: list[RobustnessProbe],
        safety_probes: list[SafetyProbe],
        perf_run: PerfRun,
        model_card: dict[str, object],
    ) -> ValidationResult:
        modules: list[ModuleResult] = []

        log.info("Stage 1/5 — faithfulness (%d records)", len(faithfulness_records))
        modules.append(get_faithfulness_evaluator(self.settings).evaluate(faithfulness_records))

        log.info("Stage 2/5 — bias (%d pairs)", len(matched_pairs))
        modules.append(BiasEvaluator().evaluate_pairs(matched_pairs))

        log.info("Stage 3/5 — robustness (%d probes)", len(robustness_probes))
        modules.append(RobustnessEvaluator().evaluate(robustness_probes))

        log.info("Stage 4/5 — safety (%d probes)", len(safety_probes))
        modules.append(SafetyEvaluator().evaluate(safety_probes))

        log.info("Stage 5/5 — performance (%d samples)", len(perf_run.samples))
        modules.append(PerformanceEvaluator().evaluate(perf_run))

        gov = completeness_score(model_card)
        log.info("Governance completeness: %.0f%%", gov * 100)

        result = self.engine.assemble(
            run_id=self.settings.run_id,
            candidate_model=self.settings.candidate_model,
            rag_pipeline=self.settings.rag_pipeline,
            config_fingerprint=self.settings.fingerprint(),
            modules=modules,
            governance_completeness=gov,
        )
        log.info("Composite risk: %s (%.2f)", result.composite_level.value,
                 result.composite_score)
        return result
