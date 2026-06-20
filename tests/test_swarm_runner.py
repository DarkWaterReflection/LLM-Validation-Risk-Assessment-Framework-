"""Tests for the concurrent swarm runner."""

from __future__ import annotations

from llm_validation_platform.config import Settings
from llm_validation_platform.orchestration.swarm_cli import build_inputs
from llm_validation_platform.orchestration.swarm_runner import (
    EVALUATOR_AGENTS,
    SwarmRunner,
)


def _run():
    return SwarmRunner(Settings(run_id="swarm-test")).run(build_inputs())


def test_swarm_runs_all_five_stages_concurrently():
    run = _run()
    stages = {t.stage for t in run.telemetry}
    assert stages == set(EVALUATOR_AGENTS)
    assert len(run.validation.modules) == 5


def test_every_stage_maps_to_an_agent():
    run = _run()
    for t in run.telemetry:
        assert t.agent_id == EVALUATOR_AGENTS[t.stage]


def test_qa_gate_passes_and_is_reproducible():
    run = _run()
    assert run.qa_passed is True
    # swarm result must match the sequential pipeline's rating
    from llm_validation_platform.orchestration.pipeline import ValidationPipeline

    seq = ValidationPipeline(Settings(run_id="swarm-test")).run(
        faithfulness_records=build_inputs().faithfulness_records,
        matched_pairs=build_inputs().matched_pairs,
        robustness_probes=build_inputs().robustness_probes,
        safety_probes=build_inputs().safety_probes,
        perf_run=build_inputs().perf_run,
        model_card=build_inputs().model_card,
    )
    assert run.validation.composite_score == seq.composite_score
    assert run.validation.composite_level == seq.composite_level


def test_telemetry_records_timing():
    run = _run()
    assert all(t.duration_ms >= 0 for t in run.telemetry)
    assert run.parallel_wall_ms >= 0
