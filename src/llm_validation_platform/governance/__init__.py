"""Governance artifacts: model cards and documentation-completeness scoring."""

from .model_card import REQUIRED_GOVERNANCE_FIELDS, completeness_score

__all__ = ["REQUIRED_GOVERNANCE_FIELDS", "completeness_score"]
