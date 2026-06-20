# LLM / RAG Model Validation & Risk Reporting Platform

A **Model Risk Management (MRM) grade** validation framework for Large Language
Models and Retrieval-Augmented Generation systems. It evaluates a model across
five risk dimensions, scores residual model risk on a documented 0–100 scale, and
renders a 17-section **Model Validation Report** suitable for risk managers,
internal audit, AI governance, and regulatory review.

Aligned to **SR 11-7**, the **NIST AI Risk Management Framework**, and the
**EU AI Act** high-risk model expectations.

## Why it's auditable

- **Deterministic core** — runs and is fully tested with **no API keys**. LLM-judge
  evaluators (RAGAS/DeepEval/Giskard) are pluggable behind the same interface.
- **Reproducible** — every run is stamped with a config fingerprint; the report is
  re-derivable from `run_id` + fingerprint.
- **Explainable** — every risk rating carries a rationale tracing to its metrics.

## Quickstart

```bash
pip install -e ".[dev]"        # core + test tooling
pytest -q                      # 23 tests, deterministic
python -m llm_validation_platform.cli --json   # generates reports/validation_report_*.md
```

Concurrent swarm run (five evaluator stages fan out across threads, mapped to
their Ruflo agent ids, with a QA reproducibility gate):

```bash
python -m llm_validation_platform.orchestration.swarm_cli
# -> reports/validation_report_*.md + artifacts/swarm_telemetry.json
```

Dashboard:

```bash
pip install -e ".[dashboard]"
streamlit run src/llm_validation_platform/dashboards/app.py
```

### Execution model (Ruflo + deterministic compute)

The Ruflo MCP layer owns **coordination**: swarm topology/state, the 10-agent
registry, shared vector memory (`namespace: mrm-validation`), and orchestration
records. The **deterministic Python `SwarmRunner`** owns *compute* — it runs the
five independent evaluator stages concurrently and is fully reproducible (the QA
gate re-runs the risk engine and asserts an identical composite). This split is
deliberate: deterministic stages must not depend on a live LLM to stay auditable.
To have the Ruflo agents perform LLM-backed execution instead, set
`ANTHROPIC_API_KEY` and call `agent_execute`, or route each stage through the
Claude Code Task tool.

### Activating the RAGAS LLM judge (faithfulness)

The faithfulness stage selects its evaluator at runtime via
`evaluations/factory.py`. RAGAS runs in-process (it makes its own judge-LLM
calls; embeddings are local sentence-transformers, so no embeddings key is
needed). To activate:

```bash
pip install -e ".[llm]"            # ragas + langchain-anthropic + HF embeddings
export ANTHROPIC_API_KEY=sk-ant-...
export MRM_USE_LLM_EVALUATORS=true # judge provider/model via MRM_JUDGE_* settings
python -m llm_validation_platform.orchestration.swarm_cli
```

Selection policy (never hard-fails):

| `use_llm_evaluators` | RAGAS installed | API key | Evaluator used |
|:---:|:---:|:---:|---|
| off | – | – | deterministic baseline |
| on | no | – | deterministic (logged fallback) |
| on | yes | no | deterministic (logged fallback) |
| on | yes | yes | **RAGAS LLM judge** |

**Effective challenge:** when the LLM judge is active, the QA gate runs the
deterministic evaluator as a cross-check oracle and flags any faithfulness
divergence beyond `CROSS_CHECK_TOLERANCE` (0.35) for human review — the LLM
judge is challenged, not blindly trusted. LLM judges are only approximately
deterministic even at temperature 0, so the deterministic baseline remains the
reproducibility anchor recorded in the config fingerprint.

## Evaluation modules

| Module | Risk dimension | Key metrics |
|---|---|---|
| `faithfulness` | Hallucination | faithfulness, context precision/recall, answer relevance, hallucination rate |
| `bias` | Bias/Fairness | fairness score, sentiment gap, disparity rate (matched-pair / counterfactual) |
| `robustness` | Robustness | semantic consistency, robustness score, instability rate |
| `safety` | Security | violation rate, refusal consistency, data-leakage count |
| `performance` | Operational | accuracy/precision/recall/F1, p95 latency, throughput, cost |
| `governance` | Governance | documentation completeness vs required artifact set |

## Risk scoring

Each dimension → 0–100 residual-risk score via documented transforms
(`risk_scoring/framework.py`), banded **Low (0–24) / Moderate (25–49) /
High (50–74) / Critical (75–100)**. Composite is a weighted average with a
**critical-override** (any Critical dimension floors the composite at High).

## Architecture

```
src/llm_validation_platform/
├── config.py            # reproducibility-first settings + fingerprint
├── schemas.py           # typed contracts shared across agents
├── evaluations/         # faithfulness, bias, robustness, safety, performance
├── risk_scoring/        # framework (bands/weights) + composite engine
├── reporting/           # Jinja2 17-section report builder + template
├── governance/          # model card + documentation-completeness scoring
├── orchestration/       # ValidationPipeline (the swarm's deterministic spine)
└── dashboards/          # Streamlit validation console
agents/swarm.yaml        # Ruflo agent-swarm topology (10 agents)
```

See `docs/` for the methodology, risk framework, and governance notes.

## Agent swarm (Ruflo)

Ten specialised agents (data-prep, four evaluators, risk-scoring, QA, reporting,
dashboard, documentation) coordinated hierarchically with a QA reproducibility
gate. Defined in `agents/swarm.yaml`; each agent calls the same deterministic
pipeline, so single-process and swarm runs are bit-for-bit equivalent.
