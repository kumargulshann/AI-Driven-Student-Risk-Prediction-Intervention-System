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

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

REPORT_PATH  = ROOT / "data" / "processed" / "risk_report.csv"
OUTCOMES_PATH = ROOT / CFG["outcome_tracking"]["outcomes_file"]

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
def load_outcomes() -> pd.DataFrame | None:
    if not OUTCOMES_PATH.exists():
        return None
    return pd.read_csv(OUTCOMES_PATH)

@st.cache_data(ttl=300)
def load_model_metrics() -> dict:
    try:
        from src.models.model_registry import load_model, model_exists
        if not model_exists():
            return {}
        _, _, _, meta = load_model()
        return meta.get("metrics", {})
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Physics_Wallah_Logo.svg/320px-Physics_Wallah_Logo.svg.png",
                 width=180)
        st.markdown("## 🎓 Early-Warning System")
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

    rec_rate = 0.0
    if outcomes_df is not None and len(outcomes_df):
        rec = outcomes_df["recovered_14d"].dropna()
        rec_rate = float(rec.mean()) if len(rec) else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Students",     f"{total:,}")
    col2.metric("🔴 Critical",         f"{critical:,}",
                delta=f"{critical/total*100:.1f}% of cohort", delta_color="inverse")
    col3.metric("🟠 High Risk",        f"{high:,}",
                delta=f"{high/total*100:.1f}% of cohort",     delta_color="inverse")
    col4.metric("⚠️  Avg Risk Score",  f"{avg_risk:.2%}")
    col5.metric("✅ Recovery Rate @14d", f"{rec_rate:.1%}")

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

    st.caption(f"Showing {len(filtered):,} students after filters")

    display_cols = ["student_id", "risk_score", "risk_tier", "batch_id",
                    "city", "course", "urgency_hours", "top_reasons"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    # Colour-coded tier column
    def color_tier(val):
        c = TIER_COLORS.get(val, "gray")
        return f"background-color: {c}20; color: {c}; font-weight: bold"

    styled = (
        filtered[display_cols]
        .sort_values("risk_score", ascending=False)
        .reset_index(drop=True)
        .style.applymap(color_tier, subset=["risk_tier"])
        .format({"risk_score": "{:.1%}"})
    )
    st.dataframe(styled, use_container_width=True, height=550)

    # Download
    csv = filtered[display_cols].to_csv(index=False)
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
    selected_id = st.selectbox("Select Student ID", student_ids)

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
    if pd.notna(row.get("top_reasons")) and str(row.get("top_reasons")).strip():
        st.markdown("#### ⚠️ Primary Risk Signals")
        for reason in str(row["top_reasons"]).split(" | "):
            if reason.strip():
                st.markdown(f"- {reason.strip()}")

    # Interventions
    if pd.notna(row.get("interventions")) and str(row.get("interventions")).strip():
        st.markdown("#### 📋 Recommended Interventions")
        for i, action in enumerate(str(row["interventions"]).split(" | "), 1):
            if action.strip():
                st.markdown(f"{i}. {action.strip()}")

    # LLM narrative
    if pd.notna(row.get("llm_narrative")) and str(row.get("llm_narrative")).strip():
        with st.expander("🤖 AI Counselor Brief"):
            st.markdown(str(row["llm_narrative"]))

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


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Intervention Tracker
# ─────────────────────────────────────────────────────────────────────────────

def page_interventions(outcomes_df: pd.DataFrame | None):
    st.title("📋 Intervention Tracker & Outcomes")

    if outcomes_df is None or len(outcomes_df) == 0:
        st.info("No intervention data yet. Run the pipeline to simulate outcomes.")
        return

    # KPIs
    total   = len(outcomes_df)
    rec_7d  = outcomes_df["recovered_7d"].dropna().mean()
    rec_14d = outcomes_df["recovered_14d"].dropna().mean()
    score_imp = outcomes_df["score_improvement"].dropna().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Interventions",      f"{total:,}")
    c2.metric("Recovery Rate @7d",        f"{rec_7d:.1%}")
    c3.metric("Recovery Rate @14d",       f"{rec_14d:.1%}")
    c4.metric("Avg Score Improvement",    f"+{score_imp:.1f} pts")

    st.markdown("---")

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Students Recovered per 1,000 Flagged")
        rec_1000 = rec_14d * 1000
        fig = go.Figure(go.Bar(
            x=["@7 days", "@14 days"],
            y=[rec_7d * 1000, rec_14d * 1000],
            marker_color=["#4299E1", "#38A169"],
            text=[f"{rec_7d*1000:.0f}", f"{rec_14d*1000:.0f}"],
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
    c4.metric("CV AUC (5-fold)",  f"{metrics.get('cv_auc_mean', 0):.3f} ± {metrics.get('cv_auc_std', 0):.3f}")

    st.markdown("---")
    st.subheader("Metrics Summary")

    metric_rows = [
        ("Precision @50%",   metrics.get("precision_at_50", 0), "0.3f"),
        ("Recall @50%",      metrics.get("recall_at_50", 0),    "0.3f"),
        ("Positive Rate",    metrics.get("positive_rate", 0),   ".1%"),
        ("Test Set Size",    metrics.get("test_size", 0),        "d"),
    ]
    for name, val, fmt in metric_rows:
        col_l, col_r = st.columns([2, 1])
        col_l.markdown(f"**{name}**")
        col_r.markdown(f"`{val:{fmt}}`")

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
        "**Imbalance handling**: `scale_pos_weight=3`\n\n"
        "**Evaluation**: Stratified 5-fold cross-validation"
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
