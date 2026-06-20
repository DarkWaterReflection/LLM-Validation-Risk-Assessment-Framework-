"""Integration tests: full pipeline run + report generation."""

from __future__ import annotations

from llm_validation_platform import sample_data
from llm_validation_platform.config import Settings
from llm_validation_platform.orchestration.pipeline import ValidationPipeline
from llm_validation_platform.reporting.report_builder import ReportBuilder
from llm_validation_platform.schemas import RiskLevel


def _result():
    pipeline = ValidationPipeline(Settings(run_id="test-run"))
    return pipeline.run(
        faithfulness_records=sample_data.faithfulness_records(),
        matched_pairs=sample_data.matched_pairs(),
        robustness_probes=sample_data.robustness_probes(),
        safety_probes=sample_data.safety_probes(),
        perf_run=sample_data.perf_run(),
        model_card=sample_data.model_card(),
    )


def test_pipeline_runs_end_to_end():
    res = _result()
    assert len(res.modules) == 5
    # governance + 5 scored dimensions
    assert len(res.dimension_risks) == 6
    assert 0 <= res.composite_score <= 100
    assert isinstance(res.composite_level, RiskLevel)


def test_sample_data_surfaces_security_risk():
    res = _result()
    sec = next(d for d in res.dimension_risks if d.dimension.value == "Security Risk")
    # sample data contains a deliberate data leak + injection compliance
    assert sec.level.ordinal >= RiskLevel.HIGH.ordinal


def test_report_contains_required_sections():
    res = _result()
    md = ReportBuilder().render_markdown(res)
    for heading in ["Executive Summary", "Final Validation Opinion",
                    "Risk Rating Framework", "Remediation Recommendations"]:
        assert heading in md
    assert res.config_fingerprint in md


def test_run_is_reproducible():
    a, b = _result(), _result()
    assert a.composite_score == b.composite_score
    assert a.config_fingerprint == b.config_fingerprint
