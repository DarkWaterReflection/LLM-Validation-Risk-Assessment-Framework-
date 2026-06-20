"""Command-line entrypoint: run a full validation and emit the report.

    python -m llm_validation_platform.cli            # uses bundled sample data
    mrm-validate --out reports                       # after `pip install -e .`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import sample_data
from .config import load_settings
from .logging_config import configure_logging
from .orchestration.pipeline import ValidationPipeline
from .reporting.report_builder import ReportBuilder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LLM/RAG model validation.")
    parser.add_argument("--out", default="reports", help="Output directory.")
    parser.add_argument("--json", action="store_true", help="Also write the raw result JSON.")
    args = parser.parse_args(argv)

    configure_logging()
    settings = load_settings()
    pipeline = ValidationPipeline(settings)

    result = pipeline.run(
        faithfulness_records=sample_data.faithfulness_records(),
        matched_pairs=sample_data.matched_pairs(),
        robustness_probes=sample_data.robustness_probes(),
        safety_probes=sample_data.safety_probes(),
        perf_run=sample_data.perf_run(),
        model_card=sample_data.model_card(),
    )

    out_dir = Path(args.out)
    report_path = ReportBuilder().write(result, out_dir)
    print(f"\nComposite Risk: {result.composite_level.value} ({result.composite_score}/100)")
    print(f"Report written: {report_path}")

    if args.json:
        json_path = out_dir / f"result_{result.run_id}.json"
        json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"Result JSON:   {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
