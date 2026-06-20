"""Typed data contracts shared across all evaluation modules.

These Pydantic models are the *interface* between agents in the swarm. Keeping
them strict and validated is what makes the pipeline auditable and reproducible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"

    @property
    def ordinal(self) -> int:
        return {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}[self.value]


class RiskDimension(str, Enum):
    HALLUCINATION = "Hallucination Risk"
    BIAS = "Bias Risk"
    ROBUSTNESS = "Robustness Risk"
    SECURITY = "Security Risk"
    OPERATIONAL = "Operational Risk"
    GOVERNANCE = "Governance Risk"


class EvalRecord(BaseModel):
    """A single question/answer/context triple under evaluation (RAG-shaped)."""

    id: str
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """One named metric with its value, valid range, and a human explanation."""

    name: str
    value: float
    higher_is_better: bool = True
    explanation: str = ""


class ModuleResult(BaseModel):
    """Aggregated output of one evaluation module (e.g. faithfulness)."""

    module: str
    dimension: RiskDimension
    metrics: list[MetricResult]
    n_records: int
    failures: list[dict] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict)

    def metric(self, name: str) -> float:
        for m in self.metrics:
            if m.name == name:
                return m.value
        raise KeyError(name)


class DimensionRisk(BaseModel):
    dimension: RiskDimension
    score: float = Field(ge=0.0, le=100.0, description="0=best, 100=worst residual risk")
    level: RiskLevel
    rationale: str
    contributing_metrics: dict[str, float] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Top-level artifact: everything needed to render the validation report."""

    run_id: str
    candidate_model: str
    rag_pipeline: str
    config_fingerprint: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    modules: list[ModuleResult] = Field(default_factory=list)
    dimension_risks: list[DimensionRisk] = Field(default_factory=list)
    composite_score: float = 0.0
    composite_level: RiskLevel = RiskLevel.MODERATE
    validation_opinion: str = ""

    def module_by_dim(self, dim: RiskDimension) -> ModuleResult | None:
        return next((m for m in self.modules if m.dimension == dim), None)
