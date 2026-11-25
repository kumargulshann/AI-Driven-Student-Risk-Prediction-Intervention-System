"""
Streamlit Counselor Dashboard
================================
Interactive UI for academic counselors to:
  1. Monitor at-risk students in real-time
  2. Deep-dive into individual student risk profiles
  3. Log interventions and track outcomes
  4. View model health and KPI metrics

Launch: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import yaml
import sys

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

with (ROOT / "config.yaml").open(encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

REPORT_PATH  = ROOT / "data" / "processed" / "risk_report.csv"

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="PW Student Early-Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
TIER_COLORS = {
    "Critical": "#E53E3E",
    "High":     "#DD6B20",
    "Medium":   "#D69E2E",
    "Low":      "#38A169",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_report() -> pd.DataFrame | None:
    if not REPORT_PATH.exists():
        return None
    df = pd.read_csv(REPORT_PATH)
    return df

@st.cache_data(ttl=60)
def load_outcomes() -> pd.DataFrame:
    from src.tracking.outcome_tracker import get_outcomes_df

    return get_outcomes_df()

@st.cache_data(ttl=300)
def load_model_metrics() -> dict:
    from src.models.model_registry import load_metadata

    try:
        return load_metadata().get("metrics", {})
    except FileNotFoundError:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("## Student Early-Warning")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["📊 Dashboard Overview",
             "🔴 At-Risk Students",
             "🔍 Student Deep-Dive",
             "📋 Intervention Tracker",
             "🤖 Model Health"],
            index=0,
        )

        st.markdown("---")
        st.markdown("**Filters**")
        tier_filter = st.multiselect(
            "Risk Tier",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"],
        )
        course_filter = st.multiselect(
            "Course",
            ["JEE Advanced", "JEE Mains", "NEET", "Foundation"],
            default=[],
        )
        min_risk = st.slider("Min Risk Score", 0.0, 1.0, 0.0, 0.05)

        st.markdown("---")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        if st.button("▶️ Run Pipeline"):
            with st.spinner("Running pipeline… this may take 2–5 minutes."):
                try:
                    from src.pipeline import run_pipeline
                    run_pipeline(train_model=True)
                    st.cache_data.clear()
                    st.success("Pipeline complete!")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

    return page, tier_filter, course_filter, min_risk


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Dashboard Overview
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame, outcomes_df: pd.DataFrame | None):
    st.title("📊 Student Early-Warning — Overview")
    st.caption(f"Showing {len(df):,} students · Snapshot day {CFG['data']['snapshot_day']}")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    total     = len(df)
    critical  = int((df["risk_tier"] == "Critical").sum())
    high      = int((df["risk_tier"] == "High").sum())
    avg_risk  = df["risk_score"].mean()

    recovery = "—"
    if outcomes_df is not None and len(outcomes_df):
        rec = outcomes_df.loc[outcomes_df["source"].eq("demo"), "recovered_14d"].dropna()
        if len(rec):
            recovery = f"{rec.mean():.1%}"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Students",     f"{total:,}")
    col2.metric("🔴 Critical",         f"{critical:,}",
                delta=f"{critical/total*100:.1f}% of cohort", delta_color="inverse")
    col3.metric("🟠 High Risk",        f"{high:,}",
                delta=f"{high/total*100:.1f}% of cohort",     delta_color="inverse")
    col4.metric("⚠️  Avg Risk Score",  f"{avg_risk:.2%}")
    col5.metric("Demo recovery @14d", recovery)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk Tier Distribution")
        tier_counts = df["risk_tier"].value_counts().reindex(
            ["Critical", "High", "Medium", "Low"], fill_value=0
        )
        fig = px.pie(
            values=tier_counts.values,
            names=tier_counts.index,
            color=tier_counts.index,
            color_discrete_map=TIER_COLORS,
            hole=0.45,
        )
        fig.update_traces(textposition="inside", textinfo="percent+value")
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Risk Score Distribution")
        fig2 = px.histogram(
            df, x="risk_score", nbins=40,
            color="risk_tier",
            color_discrete_map=TIER_COLORS,
            barmode="stack",
            labels={"risk_score": "Risk Score (probability)", "count": "Students"},
        )
        fig2.update_layout(margin=dict(t=10, b=0), legend_title="Risk Tier")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Course × Tier heatmap ─────────────────────────────────────────────────
    if "course" in df.columns:
        st.subheader("Risk Tier × Course Heatmap")
        pivot = pd.crosstab(df["course"], df["risk_tier"]).reindex(
            columns=["Critical", "High", "Medium", "Low"], fill_value=0
        )
        fig3 = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn_r",
            text_auto=True,
            aspect="auto",
        )
        fig3.update_layout(margin=dict(t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — At-Risk Students Table
# ─────────────────────────────────────────────────────────────────────────────

def page_at_risk(df: pd.DataFrame, tier_filter, course_filter, min_risk):
    st.title("🔴 At-Risk Students")

    filtered = df[df["risk_score"] >= min_risk].copy()
    if tier_filter:
        filtered = filtered[filtered["risk_tier"].isin(tier_filter)]
    if course_filter and "course" in filtered.columns:
        filtered = filtered[filtered["course"].isin(course_filter)]

    if filtered.empty:
        st.info("No students match these filters.")
        return

    display_cols = ["student_id", "risk_score", "risk_tier", "batch_id",
                    "city", "course", "urgency_hours", "top_reasons"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    # Colour-coded tier column
    def color_tier(val):
        c = TIER_COLORS.get(val, "gray")
        return f"background-color: {c}20; color: {c}; font-weight: bold"

    ranked = filtered[display_cols].sort_values("risk_score", ascending=False, kind="stable")
    page_size = 250
    pages = (len(ranked) + page_size - 1) // page_size
    page = st.number_input("Page", min_value=1, max_value=pages, step=1, disabled=pages == 1)
    start = (page - 1) * page_size
    end = min(start + page_size, len(ranked))
    st.caption(f"Students {start + 1:,}–{end:,} of {len(ranked):,} matching filters")

    styled = (ranked.iloc[start:end].style
              .map(color_tier, subset=["risk_tier"])
              .format({"risk_score": "{:.1%}"}))
    st.dataframe(
        styled, use_container_width=True, height=550, hide_index=True,
        column_config={
            "student_id": "Student",
            "risk_score": "Risk",
            "risk_tier": "Tier",
            "batch_id": "Batch",
            "city": "City",
            "course": "Course",
            "urgency_hours": st.column_config.NumberColumn("Follow-up (hours)", format="%d"),
            "top_reasons": "Top risk signals",
        },
    )

    csv = ranked.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV",
        csv,
        file_name="at_risk_students.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Student Deep-Dive
# ─────────────────────────────────────────────────────────────────────────────

def page_student_deepdive(df: pd.DataFrame):
    st.title("🔍 Student Deep-Dive")

    student_ids = sorted(df["student_id"].tolist())
    selected_id = st.selectbox("Select Student ID", student_ids, index=student_ids.index(df.iloc[0]["student_id"]))

    row = df[df["student_id"] == selected_id].iloc[0]

    # Header
    tier  = row.get("risk_tier", "Unknown")
    score = float(row.get("risk_score", 0))
    color = TIER_COLORS.get(tier, "gray")

    col1, col2, col3 = st.columns([2, 1, 1])
    col1.markdown(f"### Student {selected_id}")
    col2.markdown(
        f"<div style='background:{color}20; border-left:4px solid {color}; "
        f"padding:12px; border-radius:6px'>"
        f"<b style='color:{color}'>Risk: {score:.1%}</b><br>{tier} Tier</div>",
        unsafe_allow_html=True,
    )
    col3.metric("Urgency", f"{int(row['urgency_hours']) if pd.notna(row.get('urgency_hours')) else '—'} hrs")

    st.markdown("---")

    # Metadata
    meta_cols = ["batch_id", "city", "course"]
    meta_row = {c: row.get(c, "—") for c in meta_cols}
    st.markdown(" · ".join(f"**{k}**: {v}" for k, v in meta_row.items()))

    # Top reasons
    reasons = _split_report_list(row.get("top_reasons"))
    if reasons:
        st.markdown("#### ⚠️ Primary Risk Signals")
        for reason in reasons:
            st.markdown(f"- {reason}")

    # Interventions
    actions = _split_report_list(row.get("interventions"))
    if actions:
        st.markdown("#### 📋 Recommended Interventions")
        for i, action in enumerate(actions, 1):
            st.markdown(f"{i}. {action}")
        _show_counselor_brief(int(selected_id), score, tier, reasons, actions, row.get("llm_narrative"))
        _show_intervention_form(int(selected_id), score, tier, actions)

    # Risk score gauge
    st.markdown("#### 🎯 Risk Score Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Disengagement Probability (%)"},
        gauge={
            "axis":    {"range": [0, 100]},
            "bar":     {"color": color},
            "steps":   [
                {"range": [0, 40],   "color": "#C6F6D5"},
                {"range": [40, 65],  "color": "#FEFCBF"},
                {"range": [65, 80],  "color": "#FEEBC8"},
                {"range": [80, 100], "color": "#FED7D7"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": score * 100,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=0, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)


def _split_report_list(value) -> list[str]:
    return [item.strip() for item in value.split(" | ") if item.strip()] if isinstance(value, str) else []


def _show_counselor_brief(student_id, score, tier, reasons, actions, rule_text):
    from src.interventions.recommendation_engine import gemini_available, generate_counselor_brief

    with st.expander("Counselor brief", expanded=True):
        if isinstance(rule_text, str) and rule_text.strip():
            st.write(rule_text)
        if not gemini_available():
            st.caption("Add GEMINI_API_KEY to the project's .env file to enable optional Gemini drafting.")
            return

        case = (student_id, score, tier, tuple(reasons), tuple(actions))
        if st.button("Draft with Gemini", key="draft_counselor_brief"):
            with st.spinner("Drafting the counselor brief…"):
                brief = generate_counselor_brief(student_id, score, tier, reasons, actions)
                st.session_state["counselor_brief"] = (case, brief)

        saved = st.session_state.get("counselor_brief")
        if saved and saved[0] == case:
            brief = saved[1]
            if brief.source == "gemini":
                st.caption("Gemini draft · Review before using")
                st.write(brief.text)
            else:
                st.info("Gemini couldn't draft a brief. The rule-based brief is available above.")
                if not isinstance(rule_text, str) or not rule_text.strip():
                    st.write(brief.text)


def _show_intervention_form(student_id, score, tier, actions):
    from filelock import Timeout
    from src.tracking.outcome_tracker import log_intervention

    with st.expander("Log an intervention"):
        with st.form("log_intervention", clear_on_submit=True):
            action = st.selectbox("Action taken", actions)
            counselor = st.text_input("Mentor ID", max_chars=100)
            notes = st.text_area("Notes (optional)", max_chars=1000)
            submitted = st.form_submit_button("Save intervention")
        if not submitted:
            return
        if not counselor.strip():
            st.error("Enter a mentor ID before saving.")
            return
        try:
            log_intervention(student_id, action, score, tier, counselor.strip(), notes=notes.strip() or None)
        except (OSError, Timeout):
            st.error("The intervention could not be saved. Please try again.")
        else:
            load_outcomes.clear()
            st.success("Intervention saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Intervention Tracker
# ─────────────────────────────────────────────────────────────────────────────

def page_interventions(outcomes_df: pd.DataFrame | None):
    st.title("📋 Intervention Tracker & Outcomes")

    source = st.radio("Outcome source", ["Simulated demo", "Mentor log"], horizontal=True)
    if outcomes_df is not None:
        outcomes_df = outcomes_df.loc[outcomes_df["source"].eq("demo" if source == "Simulated demo" else "mentor")]
    if outcomes_df is None or outcomes_df.empty:
        st.info("No interventions for this view yet.")
        return

    # KPIs
    total   = len(outcomes_df)
    rec_7d  = outcomes_df["recovered_7d"].dropna().mean()
    rec_14d = outcomes_df["recovered_14d"].dropna().mean()
    score_imp = outcomes_df["score_improvement"].dropna().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Interventions",      f"{total:,}")
    c2.metric("Recovery Rate @7d",        f"{rec_7d:.1%}" if pd.notna(rec_7d) else "—")
    c3.metric("Recovery Rate @14d",       f"{rec_14d:.1%}" if pd.notna(rec_14d) else "—")
    c4.metric("Avg Score Change",        f"{score_imp:+.1f} pts" if pd.notna(score_imp) else "—")

    st.markdown("---")

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Recovery per 1,000 Follow-ups")
        fig = go.Figure(go.Bar(
            x=["@7 days", "@14 days"],
            y=[rec_7d * 1000, rec_14d * 1000],
            marker_color=["#4299E1", "#38A169"],
            text=[f"{rate * 1000:.0f}" if pd.notna(rate) else "" for rate in (rec_7d, rec_14d)],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="Recovered per 1,000",
            margin=dict(t=20, b=0),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.subheader("Risk Score: Before vs After Intervention")
        scatter_df = outcomes_df.dropna(subset=["risk_score_before", "risk_score_14d"])
        if len(scatter_df):
            fig2 = px.scatter(
                scatter_df,
                x="risk_score_before", y="risk_score_14d",
                color="risk_tier_before",
                color_discrete_map=TIER_COLORS,
                labels={"risk_score_before": "Risk Score Before", "risk_score_14d": "Risk Score @14d"},
                opacity=0.6,
            )
            # Reference line y=x (no change)
            fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                           line=dict(color="gray", dash="dash"))
            fig2.update_layout(height=300, margin=dict(t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

    # Outcomes table
    st.subheader("Intervention Log")
    st.dataframe(
        outcomes_df[["student_id", "intervention_type", "risk_score_before",
                     "risk_score_7d", "risk_score_14d", "recovered_7d",
                     "recovered_14d", "score_improvement", "counselor_id"]]
        .sort_values("risk_score_before", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=400,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — Model Health
# ─────────────────────────────────────────────────────────────────────────────

def page_model_health():
    st.title("🤖 Model Health & Performance")

    metrics = load_model_metrics()
    if not metrics:
        st.warning("Model not trained yet. Run the pipeline from the sidebar.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC",          f"{metrics.get('auc_roc', 0):.3f}")
    c2.metric("Avg Precision",    f"{metrics.get('avg_precision', 0):.3f}")
    c3.metric("F1 Score @50%",    f"{metrics.get('f1_at_50', 0):.3f}")
    cv_auc = f"{metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}" if "cv_auc_mean" in metrics else "Unavailable"
    c4.metric("CV AUC (5-fold)", cv_auc)

    st.markdown("---")
    st.subheader("Metrics Summary")

    metric_rows = [
        ("Precision @50%",   metrics.get("precision_at_50", 0), "0.3f"),
        ("Recall @50%",      metrics.get("recall_at_50", 0),    "0.3f"),
        ("Brier score before calibration", metrics.get("brier_before"), "0.3f"),
        ("Brier score after calibration", metrics.get("brier_after"), "0.3f"),
        ("Positive Rate",    metrics.get("positive_rate", 0),   ".1%"),
        ("Test Set Size",    metrics.get("test_size", 0),        "d"),
    ]
    for name, val, fmt in metric_rows:
        col_l, col_r = st.columns([2, 1])
        col_l.markdown(f"**{name}**")
        col_r.markdown("Unavailable" if val is None else f"`{val:{fmt}}`")

    st.markdown("---")
    st.subheader("Risk Tier Thresholds")
    thresholds = CFG["risk_tiers"]
    tier_df = pd.DataFrame([
        {"Tier": k.title(), "Min Score": v[0], "Max Score": v[1]}
        for k, v in thresholds.items()
    ])
    st.table(tier_df)

    st.info(
        "💡 **Model**: LightGBM + Platt Scaling Calibration\n\n"
        f"**Imbalance handling**: `scale_pos_weight={CFG['model']['scale_pos_weight']}`\n\n"
        "**Evaluation**: Stratified cross-validation on the development set, plus a held-out test set"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    page, tier_filter, course_filter, min_risk = sidebar()

    df = load_report()
    outcomes_df = load_outcomes()

    if df is None:
        st.warning(
            "⚠️ No risk report found.\n\n"
            "Click **▶️ Run Pipeline** in the sidebar to generate one (takes ~3–5 minutes)."
        )
        if st.button("▶️ Run Pipeline Now"):
            with st.spinner("Running pipeline…"):
                from src.pipeline import run_pipeline
                run_pipeline(train_model=True)
                st.cache_data.clear()
                st.rerun()
        return

    if page == "📊 Dashboard Overview":
        page_overview(df, outcomes_df)
    elif page == "🔴 At-Risk Students":
        page_at_risk(df, tier_filter, course_filter, min_risk)
    elif page == "🔍 Student Deep-Dive":
        page_student_deepdive(df)
    elif page == "📋 Intervention Tracker":
        page_interventions(outcomes_df)
    elif page == "🤖 Model Health":
        page_model_health()


if __name__ == "__main__":
    main()
