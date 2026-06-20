"""Concurrent swarm execution of the validation DAG.

This is the executable counterpart to ``agents/swarm.yaml``. The five evaluation
modules are mutually independent (they all consume the data-prep output and
nothing else), so they fan out across a thread pool — exactly how the Ruflo swarm
schedules the faithfulness/bias/robustness/safety/performance agents in parallel.
The downstream stages (risk-scoring, QA gate, reporting) are dependency-ordered.

Each stage is annotated with the Ruflo agent id that owns it, plus wall-clock
telemetry, so the orchestration layer can push per-agent evidence into the shared
swarm memory and prove the run was reproducible.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from ..config import Settings
from ..evaluations.bias import BiasEvaluator, MatchedPair
from ..evaluations.factory import cross_check_faithfulness, get_faithfulness_evaluator
from ..evaluations.performance import PerfRun, PerformanceEvaluator
from ..evaluations.robustness import RobustnessEvaluator, RobustnessProbe
from ..evaluations.safety import SafetyEvaluator, SafetyProbe
from ..governance.model_card import completeness_score
from ..logging_config import get_logger
from ..risk_scoring.engine import RiskEngine
from ..schemas import EvalRecord, ModuleResult, ValidationResult

log = get_logger("swarm")

# Maps each parallel evaluation stage to the Ruflo agent that owns it.
EVALUATOR_AGENTS = {
    "faithfulness": "faithfulness-agent",
    "bias": "bias-agent",
    "robustness": "robustness-agent",
    "safety": "safety-agent",
    "performance": "documentation-agent",  # perf telemetry owned alongside docs
}


@dataclass
class StageTelemetry:
    stage: str
    agent_id: str
    duration_ms: float
    started_at: float
    n_records: int = 0
    note: str = ""


@dataclass
class SwarmRunResult:
    validation: ValidationResult
    telemetry: list[StageTelemetry] = field(default_factory=list)
    qa_passed: bool = False
    qa_note: str = ""
    parallel_wall_ms: float = 0.0
    parallel_serial_ms: float = 0.0

    @property
    def speedup(self) -> float:
        return self.parallel_serial_ms / self.parallel_wall_ms if self.parallel_wall_ms else 1.0


@dataclass
class SwarmInputs:
    faithfulness_records: list[EvalRecord]
    matched_pairs: list[MatchedPair]
    robustness_probes: list[RobustnessProbe]
    safety_probes: list[SafetyProbe]
    perf_run: PerfRun
    model_card: dict[str, object]


class SwarmRunner:
    """Executes the validation DAG with a concurrent evaluator fan-out."""

    def __init__(self, settings: Settings, max_workers: int = 5) -> None:
        self.settings = settings
        self.engine = RiskEngine()
        self.max_workers = max_workers

    def run(self, inputs: SwarmInputs) -> SwarmRunResult:
        telemetry: list[StageTelemetry] = []

        # --- Fan-out: independent evaluator agents run concurrently -------
        faithfulness_evaluator = get_faithfulness_evaluator(self.settings)
        tasks: dict[str, Callable[[], ModuleResult]] = {
            "faithfulness": lambda: faithfulness_evaluator.evaluate(inputs.faithfulness_records),
            "bias": lambda: BiasEvaluator().evaluate_pairs(inputs.matched_pairs),
            "robustness": lambda: RobustnessEvaluator().evaluate(inputs.robustness_probes),
            "safety": lambda: SafetyEvaluator().evaluate(inputs.safety_probes),
            "performance": lambda: PerformanceEvaluator().evaluate(inputs.perf_run),
        }

        modules: dict[str, ModuleResult] = {}
        serial_ms = 0.0
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._timed, name, fn): name for name, fn in tasks.items()}
            for fut in as_completed(futures):
                name = futures[fut]
                result, dur = fut.result()
                modules[name] = result
                serial_ms += dur
                telemetry.append(
                    StageTelemetry(
                        stage=name,
                        agent_id=EVALUATOR_AGENTS[name],
                        duration_ms=round(dur, 3),
                        started_at=wall_start,
                        n_records=result.n_records,
                        note=f"{len(result.failures)} finding(s)",
                    )
                )
                log.info("agent %-22s completed %-12s in %.2fms",
                         EVALUATOR_AGENTS[name], name, dur)
        wall_ms = (time.perf_counter() - wall_start) * 1000

        # Preserve canonical module order for the report.
        ordered = [modules[n] for n in
                   ("faithfulness", "bias", "robustness", "safety", "performance")]

        # --- risk-scoring-agent ------------------------------------------
        gov = completeness_score(inputs.model_card)
        validation = self.engine.assemble(
            run_id=self.settings.run_id,
            candidate_model=self.settings.candidate_model,
            rag_pipeline=self.settings.rag_pipeline,
            config_fingerprint=self.settings.fingerprint(),
            modules=ordered,
            governance_completeness=gov,
        )

        # --- validation-qa-agent: reproducibility gate -------------------
        qa_passed, qa_note = self._qa_gate(ordered, gov, validation)

        # Effective challenge of the LLM judge via the deterministic oracle.
        if self.settings.use_llm_evaluators:
            faithfulness_module = ordered[0]
            within_tol, cc_note = cross_check_faithfulness(
                inputs.faithfulness_records, faithfulness_module
            )
            qa_passed = qa_passed and within_tol
            qa_note = f"{qa_note} | LLM-judge cross-check: {cc_note}"

        return SwarmRunResult(
            validation=validation,
            telemetry=sorted(telemetry, key=lambda t: t.stage),
            qa_passed=qa_passed,
            qa_note=qa_note,
            parallel_wall_ms=round(wall_ms, 3),
            parallel_serial_ms=round(serial_ms, 3),
        )

    def _qa_gate(self, modules, gov, original) -> tuple[bool, str]:
        """Independently re-run the engine; the composite must match exactly."""
        replay = RiskEngine().assemble(
            run_id=self.settings.run_id,
            candidate_model=self.settings.candidate_model,
            rag_pipeline=self.settings.rag_pipeline,
            config_fingerprint=self.settings.fingerprint(),
            modules=modules,
            governance_completeness=gov,
        )
        ok = (
            replay.composite_score == original.composite_score
            and replay.composite_level == original.composite_level
            and replay.config_fingerprint == original.config_fingerprint
        )
        note = (
            f"Reproduced composite {replay.composite_score}/100 "
            f"({replay.composite_level.value}); fingerprint match."
            if ok else "QA FAILED: replay diverged from original run."
        )
        return ok, note

    @staticmethod
    def _timed(name: str, fn: Callable[[], ModuleResult]) -> tuple[ModuleResult, float]:
        start = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - start) * 1000
