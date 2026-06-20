"""Run the validation as a concurrent agent swarm and emit telemetry.

    python -m llm_validation_platform.orchestration.swarm_cli

Prints a JSON envelope (per-agent stage telemetry + composite rating + QA gate)
suitable for ingestion into the Ruflo shared swarm memory, and writes the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import sample_data
from ..config import load_settings
from ..logging_config import configure_logging
from ..reporting.report_builder import ReportBuilder
from .swarm_runner import SwarmInputs, SwarmRunner


def build_inputs() -> SwarmInputs:
    return SwarmInputs(
        faithfulness_records=sample_data.faithfulness_records(),
        matched_pairs=sample_data.matched_pairs(),
        robustness_probes=sample_data.robustness_probes(),
        safety_probes=sample_data.safety_probes(),
        perf_run=sample_data.perf_run(),
        model_card=sample_data.model_card(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run validation as a concurrent swarm.")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--telemetry", default="artifacts/swarm_telemetry.json")
    args = parser.parse_args(argv)

    configure_logging()
    settings = load_settings()
    run = SwarmRunner(settings).run(build_inputs())

    ReportBuilder().write(run.validation, Path(args.out))

    envelope = {
        "swarm": "mrm-llm-validation",
        "run_id": run.validation.run_id,
        "config_fingerprint": run.validation.config_fingerprint,
        "composite_score": run.validation.composite_score,
        "composite_level": run.validation.composite_level.value,
        "validation_opinion": run.validation.validation_opinion,
        "qa_passed": run.qa_passed,
        "qa_note": run.qa_note,
        "parallel_wall_ms": run.parallel_wall_ms,
        "serial_equivalent_ms": run.parallel_serial_ms,
        "concurrency_speedup": round(run.speedup, 2),
        "dimension_risks": [
            {"dimension": d.dimension.value, "score": d.score, "level": d.level.value}
            for d in run.validation.dimension_risks
        ],
        "agent_telemetry": [
            {
                "agent_id": t.agent_id,
                "stage": t.stage,
                "duration_ms": t.duration_ms,
                "n_records": t.n_records,
                "note": t.note,
            }
            for t in run.telemetry
        ],
    }

    tel_path = Path(args.telemetry)
    tel_path.parent.mkdir(parents=True, exist_ok=True)
    tel_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    print(json.dumps(envelope, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
