"""Streamlit validation dashboard.

Run:  streamlit run src/llm_validation_platform/dashboards/app.py

Lets a reviewer execute the validation on bundled sample data (or uploaded CSVs),
inspect per-dimension risk, drill into failures, and download the report. Heavy
viz libs (plotly) are imported lazily so the core platform never depends on them.
"""

from __future__ import annotations

import streamlit as st

from llm_validation_platform import sample_data
from llm_validation_platform.config import load_settings
from llm_validation_platform.orchestration.pipeline import ValidationPipeline
from llm_validation_platform.reporting.report_builder import ReportBuilder

_LEVEL_COLOR = {"Low": "#2e7d32", "Moderate": "#f9a825", "High": "#ef6c00", "Critical": "#c62828"}


def _run():
    settings = load_settings()
    return ValidationPipeline(settings).run(
        faithfulness_records=sample_data.faithfulness_records(),
        matched_pairs=sample_data.matched_pairs(),
        robustness_probes=sample_data.robustness_probes(),
        safety_probes=sample_data.safety_probes(),
        perf_run=sample_data.perf_run(),
        model_card=sample_data.model_card(),
    )


def main() -> None:
    st.set_page_config(page_title="LLM Model Validation & Risk", layout="wide")
    st.title("🛡️ LLM / RAG Model Validation & Risk Reporting")
    st.caption("Model Risk Management Group — independent validation console")

    if st.sidebar.button("▶ Run validation (sample data)", type="primary"):
        st.session_state["result"] = _run()

    result = st.session_state.get("result")
    if result is None:
        st.info("Use the sidebar to run a validation on the bundled sample dataset.")
        return

    color = _LEVEL_COLOR.get(result.composite_level.value, "#555")
    st.markdown(
        f"### Composite Risk: "
        f"<span style='color:{color}'>**{result.composite_level.value}** "
        f"({result.composite_score}/100)</span>",
        unsafe_allow_html=True,
    )
    st.write(result.validation_opinion)

    cols = st.columns(len(result.dimension_risks))
    for col, d in zip(cols, result.dimension_risks):
        col.metric(d.dimension.value, f"{d.score:.0f}", d.level.value)

    st.subheader("Risk heatmap")
    try:
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(
            {"dimension": [d.dimension.value for d in result.dimension_risks],
             "score": [d.score for d in result.dimension_risks]}
        )
        fig = px.bar(df, x="score", y="dimension", orientation="h",
                     range_x=[0, 100], color="score",
                     color_continuous_scale=["#2e7d32", "#f9a825", "#c62828"])
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart({d.dimension.value: d.score for d in result.dimension_risks})

    st.subheader("Findings")
    for d in result.dimension_risks:
        mod = result.module_by_dim(d.dimension)
        with st.expander(f"{d.dimension.value} — {d.level.value} ({d.score})"):
            st.write(d.rationale)
            if mod and mod.failures:
                st.json(mod.failures)

    st.subheader("Validation report")
    md = ReportBuilder().render_markdown(result)
    st.download_button("⬇ Download report (Markdown)", md,
                       file_name=f"validation_report_{result.run_id}.md")
    with st.expander("Preview report"):
        st.markdown(md)


if __name__ == "__main__":
    main()
