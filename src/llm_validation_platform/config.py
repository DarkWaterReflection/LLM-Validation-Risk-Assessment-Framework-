"""Centralised, reproducibility-first configuration.

All run-affecting parameters (seeds, thresholds, model identifiers) are captured
here so that a validation run can be reproduced exactly and the configuration can
be hashed into the report's audit trail.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Run-wide settings. Override via env vars prefixed ``MRM_`` or a .env file."""

    model_config = SettingsConfigDict(env_prefix="MRM_", env_file=".env", extra="ignore")

    # --- Reproducibility -------------------------------------------------
    random_seed: int = 42
    run_id: str = Field(default="local-run", description="Logical id for this validation run.")

    # --- System under validation ----------------------------------------
    candidate_model: str = "candidate-llm"
    rag_pipeline: str = "baseline-rag"

    # --- Provider wiring (optional; engine runs without it) --------------
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    judge_model: str = "claude-opus-4-8"
    judge_provider: str = "anthropic"  # "anthropic" | "openai"
    judge_temperature: float = 0.0     # 0 for max determinism of the LLM judge
    # Local sentence-transformers embeddings => no embeddings API key required.
    judge_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Paths -----------------------------------------------------------
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")

    # --- Evaluation toggles ---------------------------------------------
    use_llm_evaluators: bool = False  # False => deterministic heuristic baselines

    def fingerprint(self) -> str:
        """Stable hash of run-affecting settings for the report audit trail."""
        payload = {
            "random_seed": self.random_seed,
            "candidate_model": self.candidate_model,
            "rag_pipeline": self.rag_pipeline,
            "judge_model": self.judge_model,
            "judge_provider": self.judge_provider,
            "judge_temperature": self.judge_temperature,
            "judge_embedding_model": self.judge_embedding_model,
            "use_llm_evaluators": self.use_llm_evaluators,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_settings() -> Settings:
    """Factory used across the platform (eases test injection)."""
    return Settings()
