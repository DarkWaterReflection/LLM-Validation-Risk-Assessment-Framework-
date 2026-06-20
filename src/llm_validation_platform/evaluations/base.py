"""Common evaluator interface + lightweight, dependency-free NLP helpers.

The deterministic helpers here let the entire platform run and be unit-tested
*without* network access or API keys. When ``Settings.use_llm_evaluators`` is on,
each module swaps in a RAGAS/DeepEval/Giskard-backed implementation behind the
same interface, so the orchestration and reporting layers never change.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter

from ..schemas import EvalRecord, ModuleResult

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cosine_bow(a: str, b: str) -> float:
    """Bag-of-words cosine similarity — a deterministic stand-in for embeddings."""
    ca, cb = Counter(tokenize(a)), Counter(tokenize(b))
    if not ca or not cb:
        return 0.0
    common = set(ca) & set(cb)
    dot = sum(ca[t] * cb[t] for t in common)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def context_support(answer: str, contexts: list[str]) -> float:
    """Fraction of answer content tokens supported by the retrieved context.

    A transparent proxy for RAGAS faithfulness: low support => potential
    unsupported (hallucinated) claim.
    """
    ans_tokens = [t for t in token_set(answer) if len(t) > 2]
    if not ans_tokens:
        return 1.0
    ctx_tokens: set[str] = set()
    for c in contexts:
        ctx_tokens |= token_set(c)
    supported = sum(1 for t in ans_tokens if t in ctx_tokens)
    return supported / len(ans_tokens)


class Evaluator(ABC):
    """Base class for every evaluation module."""

    name: str = "evaluator"

    @abstractmethod
    def evaluate(self, records: list[EvalRecord]) -> ModuleResult: ...
