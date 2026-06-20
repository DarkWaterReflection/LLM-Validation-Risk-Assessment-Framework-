"""Governance & documentation-risk scoring.

The governance dimension is scored from the completeness of the required model
documentation set — the same discipline a validator applies when checking whether
a model owner's package meets policy (model card, intended use, limitations,
monitoring plan, owner accountability, etc.).
"""

from __future__ import annotations

# Required documentation artifacts for an LLM/RAG model under MRM policy.
REQUIRED_GOVERNANCE_FIELDS: tuple[str, ...] = (
    "model_name",
    "model_owner",
    "intended_use",
    "training_data_description",
    "limitations",
    "performance_metrics",
    "monitoring_plan",
    "human_oversight",
    "data_privacy_assessment",
    "approval_record",
)


def completeness_score(model_card: dict[str, object]) -> float:
    """Share of required fields that are present and non-empty (0.0–1.0)."""
    present = sum(
        1 for f in REQUIRED_GOVERNANCE_FIELDS
        if str(model_card.get(f, "")).strip()
    )
    return present / len(REQUIRED_GOVERNANCE_FIELDS)


def missing_fields(model_card: dict[str, object]) -> list[str]:
    return [f for f in REQUIRED_GOVERNANCE_FIELDS if not str(model_card.get(f, "")).strip()]
