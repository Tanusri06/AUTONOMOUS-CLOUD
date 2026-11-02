
from pathlib import Path
import sys
import uuid
import json
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PATHS AND IMPORTS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
RESULTS_ROOT = ROOT / "outputs" / "experiment_results"
UPLOAD_RESULTS_ROOT = RESULTS_ROOT / "uploaded_experiments"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prepare_simulation import prepare_uploaded_workload
from simulator import run_experiment

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aegis Cloud | Multi-Cloud Orchestration",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: #101e31;
        --panel2: #14243a;
        --border: #263b55;
        --text: #e7eef8;
        --muted: #9db0c8;
        --blue: #60a5fa;
        --cyan: #5eead4;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(37,99,235,.14), transparent 30%),
            radial-gradient(circle at 90% 10%, rgba(20,184,166,.09), transparent 25%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7,17,31,.85);
    }

    [data-testid="stSidebar"] {
        background: #091626;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .hero {
        padding: 2.2rem 2.4rem;
        border: 1px solid #294563;
        border-radius: 24px;
        background:
            linear-gradient(120deg, rgba(20,42,69,.96), rgba(10,25,43,.94)),
            #101e31;
        margin-bottom: 1.5rem;
    }

    .hero-chip {
        display: inline-block;
        border: 1px solid #315b7b;
        background: rgba(30,64,94,.55);
        color: #a5e8ff;
        padding: .35rem .75rem;
        border-radius: 100px;
        font-size: .78rem;
        letter-spacing: .04em;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: clamp(2.1rem, 4vw, 3.5rem);
        line-height: 1.08;
        font-weight: 800;
        letter-spacing: -.04em;
        color: #f1f6ff;
        margin: .3rem 0 1rem 0;
    }

    .hero-copy {
        max-width: 780px;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #b6c7dc;
    }

    .section-kicker {
        color: #60a5fa;
        font-size: .78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .13em;
        margin-bottom: .35rem;
    }

    .section-heading {
        color: #f1f6ff;
        font-size: 1.55rem;
        font-weight: 750;
        margin-bottom: .35rem;
    }

    .section-copy {
        color: #9db0c8;
        margin-bottom: 1.2rem;
    }

    .feature-card {
        background: linear-gradient(145deg, #12233a, #0d1a2b);
        border: 1px solid #263e5a;
        border-radius: 18px;
        padding: 1.25rem;
        min-height: 205px;
        margin-bottom: .6rem;
    }

    .feature-icon {
        font-size: 1.7rem;
        margin-bottom: .7rem;
    }

    .feature-title {
        color: #eaf2ff;
        font-size: 1.1rem;
        font-weight: 750;
        margin-bottom: .45rem;
    }

    .feature-copy {
        color: #9db0c8;
        font-size: .92rem;
        line-height: 1.55;
    }

    .status-pill {
        display: inline-block;
        border-radius: 100px;
        padding: .4rem .8rem;
        border: 1px solid #285b54;
        background: rgba(13,75,65,.3);
        color: #8ef0d6;
        font-size: .82rem;
        font-weight: 650;
    }

    .small-muted {
        color: #91a6bf;
        font-size: .86rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #13243a, #0d1b2c);
        border: 1px solid #29415d;
        padding: 1rem;
        border-radius: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9db0c8;
    }

    div[data-testid="stMetricValue"] {
        color: #eaf2ff;
    }

    div[data-testid="stMetricDelta"] {
        color: #5eead4;
    }

    /* ============================================================
       NORMAL BUTTONS — DARK BACKGROUND, READABLE TEXT
       ============================================================ */

    div.stButton > button,
    div.stDownloadButton > button {
        background-color: #18324a !important;
        color: #ffffff !important;
        border: 1px solid #3b6688 !important;
        border-radius: 10px !important;
        min-height: 2.7rem;
        font-weight: 650;
    }

    div.stButton > button *,
    div.stDownloadButton > button * {
        color: #ffffff !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background-color: #244967 !important;
        color: #ffffff !important;
        border: 1px solid #6da9d6 !important;
    }

    /* Keep primary action button blue/teal */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb, #0d9488) !important;
        border: none !important;
        color: #ffffff !important;
    }

    div.stButton > button[kind="primary"] * {
        color: #ffffff !important;
    }

    /* ============================================================
       FILE UPLOADER
       ============================================================ */

    div[data-testid="stFileUploader"] {
        border: 1px dashed #42617f;
        border-radius: 14px;
        padding: .8rem;
        background: rgba(15,31,50,.55);
    }

    div[data-testid="stFileUploader"] button {
        background-color: #18324a !important;
        color: #ffffff !important;
        border: 1px solid #3b6688 !important;
    }

    div[data-testid="stFileUploader"] button * {
        color: #ffffff !important;
    }

    div[data-testid="stFileUploader"] section {
        background-color: #101e31 !important;
    }

    div[data-testid="stFileUploader"] section * {
        color: #dce8f7 !important;
    }

    hr {
        border-color: #263b55;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "page": "Home",
    "experiment": None,
    "uploaded_filename": None,
    "experiment_id": None,
    "experiment_dir": None,
    "last_error": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def navigate(page_name):
    st.session_state.page = page_name


# ============================================================
# HELPERS
# ============================================================

def format_metric(value, suffix="", decimals=2):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}{suffix}"


def get_policy_value(summary, policy, column):
    if summary.empty or column not in summary.columns:
        return None

    rows = summary[summary["policy"].astype(str) == policy]
    if rows.empty:
        return None

    return rows.iloc[0][column]


def download_df(data, filename, label):
    if data is None or data.empty:
        st.caption("No data available for this download.")
        return

    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def chart_layout(fig, height=390):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dce8f7",
        title_font_color="#eaf2ff",
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        legend_title_text="",
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,.12)",
        linecolor="#344a63",
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,.12)",
        linecolor="#344a63",
    )
    return fig


def current_experiment():
    return st.session_state.experiment


def run_uploaded_experiment(uploaded_file):
    if uploaded_file is None:
        st.error("Upload a CSV file from your laptop before running.")
        return

    experiment_id = uuid.uuid4().hex[:10]
    output_dir = UPLOAD_RESULTS_ROOT / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    uploaded_copy = output_dir / uploaded_file.name

    try:
        with st.spinner(
            "Validating workload, running ML analysis, and comparing policies..."
        ):
            uploaded_copy.write_bytes(uploaded_file.getvalue())

            uploaded_df = pd.read_csv(uploaded_copy)

            # prepare_uploaded_workload expects a DataFrame.
            workload, failures = prepare_uploaded_workload(uploaded_df)

            summary, observations, actions, capacity = run_experiment(
                workload=workload,
                failures=failures,
                output_dir=output_dir,
            )

            # Keep the current run's data in session state.
            st.session_state.experiment = {
                "summary": summary.copy(),
                "observations": observations.copy(),
                "actions": actions.copy(),
                "capacity": capacity.copy(),
            }

            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.experiment_id = experiment_id
            st.session_state.experiment_dir = str(output_dir)
            st.session_state.last_error = None
            st.session_state.page = "Overview"

        st.success("Experiment completed successfully.")

    except Exception as error:
        st.session_state.last_error = str(error)
        st.error(f"Experiment failed: {error}")
        st.code(str(error))


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

has_results = current_experiment() is not None

st.sidebar.markdown("## ☁️ Aegis Cloud")
st.sidebar.caption("Autonomous Multi-Cloud Orchestration")
st.sidebar.divider()

nav_pages = ["Home", "About", "How It Works"]

if has_results:
    nav_pages += [
        "Overview",
        "Workload & ML",
        "Orchestration Results",
        "SLA & Self-Healing",
        "Cloud Utilization",
        "Action Log",
    ]

selected_page = st.sidebar.radio(
    "Navigate",
    nav_pages,
    index=nav_pages.index(st.session_state.page)
    if st.session_state.page in nav_pages
    else 0,
)

st.session_state.page = selected_page

st.sidebar.divider()
st.sidebar.markdown("### Experiment Workspace")
st.sidebar.caption("Upload a workload CSV from your computer.")

sidebar_upload = st.sidebar.file_uploader(
    "Choose workload CSV",
    type=["csv"],
    key="sidebar_upload",
    help="Expected: 30 VMs × 288 five-minute records per VM.",
)

if st.sidebar.button(
    "▶ Run Experiment",
    type="primary",
    use_container_width=True,
):
    run_uploaded_experiment(sidebar_upload)

if has_results:
    st.sidebar.success("Current experiment is ready.")
    st.sidebar.caption(f"File: {st.session_state.uploaded_filename}")
    st.sidebar.caption(f"Run ID: {st.session_state.experiment_id}")

    if st.sidebar.button("Clear Current Results", use_container_width=True):
        st.session_state.experiment = None
        st.session_state.uploaded_filename = None
        st.session_state.experiment_id = None
        st.session_state.experiment_dir = None
        st.session_state.page = "Home"
        st.rerun()

st.sidebar.caption(
    "Research prototype · Simulated cloud orchestration · "
    "No live cloud resources are provisioned"
)

# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-chip">ML-assisted infrastructure intelligence</div>
            <div class="small-muted">RESEARCH PROTOTYPE · MULTI-CLOUD SYSTEMS</div>
            <div class="hero-title">
                Orchestrate smarter.<br>
                Recover faster.
            </div>
            <div class="hero-copy">
                Explore workload prediction, anomaly detection, multi-cloud
                orchestration, and simulated self-healing through an interactive
                infrastructure research platform.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">Start here</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Explore the platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">'
        'Choose a module. You can return to Home at any time using the sidebar.'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🧪</div>
                <div class="feature-title">Run an Experiment</div>
                <div class="feature-copy">
                    Upload a workload CSV from your laptop and compare
                    Static, Reactive, and Risk-Aware orchestration policies.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        home_upload = st.file_uploader(
            "Upload workload CSV",
            type=["csv"],
            key="home_upload",
        )
        if st.button("Run uploaded workload", type="primary", use_container_width=True):
            run_uploaded_experiment(home_upload)

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">☁️</div>
                <div class="feature-title">About the Platform</div>
                <div class="feature-copy">
                    Understand the research objective, architecture,
                    technologies, policies, and project scope.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore the platform", use_container_width=True):
            navigate("About")
            st.rerun()

    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⚙️</div>
                <div class="feature-title">How It Works</div>
                <div class="feature-copy">
                    Follow the workflow from workload ingestion and ML
                    analysis through orchestration, recovery, and evaluation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View system workflow", use_container_width=True):
            navigate("How It Works")
            st.rerun()

    st.divider()

    st.markdown('<div class="section-kicker">Platform capabilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">What the platform evaluates</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    f4, f5, f6 = st.columns(3)

    with f1:
        st.markdown("**📈 CPU workload prediction**")
        st.caption("Random Forest predicts CPU utilization.")

    with f2:
        st.markdown("**🔎 Anomaly detection**")
        st.caption("Isolation Forest identifies unusual workload behavior.")

    with f3:
        st.markdown("**🔁 Policy comparison**")
        st.caption("Static, Reactive, and Risk-Aware policies.")

    with f4:
        st.markdown("**🛟 Simulated recovery**")
        st.caption("Cloud failures and migration responses are simulated.")

    with f5:
        st.markdown("**📊 SLA and capacity analysis**")
        st.caption("Availability, SLA violations, recovery, and utilization.")

    with f6:
        st.markdown("**📥 Experiment exports**")
        st.caption("Download the current run's result CSV files.")

    if has_results:
        st.divider()
        st.markdown(
            f'<span class="status-pill">Current run ready · {st.session_state.experiment_id}</span>',
            unsafe_allow_html=True,
        )
        if st.button("Open current experiment results", type="primary"):
            navigate("Overview")
            st.rerun()

# ============================================================
# ABOUT PAGE
# ============================================================

elif st.session_state.page == "About":

    st.markdown('<div class="section-kicker">Project information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">About Aegis Cloud</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### Autonomous Multi-Cloud Resource Orchestration and Self-Healing

        **Research objective:** Compare orchestration policies and evaluate
        how machine-learning-assisted workload analysis and failure-aware
        decisions affect service continuity.

        ### Core components

        1. Workload preprocessing and validation
        2. Random Forest CPU prediction
        3. Isolation Forest anomaly detection
        4. Multi-cloud orchestration simulator
        5. Simulated failure detection and VM migration
        6. SLA, recovery, utilization, and illustrative cost analysis
        7. Interactive Streamlit dashboard

        ### Orchestration policies

        - **Static:** retains the initial VM placement.
        - **Reactive:** responds to cloud failures and CPU overload.
        - **Risk-Aware:** considers failure signals, risk, anomalies, CPU
          behavior, latency, and projected cloud capacity.

        ### Technologies

        Python, Pandas, NumPy, Scikit-learn, Joblib, Streamlit, Plotly,
        and Matplotlib.

        ### Scope and limitations

        The platform uses workload data and saved ML models. The multi-cloud
        environment, failure response, recovery, and migration are simulated.
        It does not provision or migrate live VMs between cloud providers.
        Cost values are illustrative and are not actual provider billing.
        """
    )

    st.divider()
    if st.button("← Back to Home"):
        navigate("Home")
        st.rerun()

# ============================================================
# HOW IT WORKS PAGE
# ============================================================

elif st.session_state.page == "How It Works":

    st.markdown('<div class="section-kicker">System workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">How the platform works</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Workload ingestion", "Upload a standardized CSV containing VM telemetry at five-minute intervals."),
        ("02", "Validation and preprocessing", "Check required columns, timestamps, VM count, and records per VM."),
        ("03", "ML analysis", "Random Forest predicts CPU utilization; Isolation Forest flags anomalies."),
        ("04", "Policy evaluation", "Run Static, Reactive, and Risk-Aware policies against the same workload."),
        ("05", "Simulated self-healing", "Evaluate failures, capacity, and migration actions in the simulated clouds."),
        ("06", "Results and export", "Compare availability, SLA violations, migrations, recovery, utilization, and cost."),
    ]

    for number, title, description in steps:
        st.markdown(
            f"""
            <div class="feature-card" style="min-height:unset; display:flex; gap:1rem; align-items:flex-start;">
                <div style="font-size:1.35rem; font-weight:800; color:#5eead4; min-width:2.5rem;">
                    {number}
                </div>
                <div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-copy">{description}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Workflow summary")
    st.code(
        "Upload → Validate → Predict CPU → Detect Anomalies → "
        "Assess Risk → Orchestrate → Simulate Recovery → Evaluate",
        language="text",
    )

    if st.button("← Back to Home"):
        navigate("Home")
        st.rerun()

# ============================================================
# RESULTS PAGES
# ============================================================

elif st.session_state.page in [
    "Overview",
    "Workload & ML",
    "Orchestration Results",
    "SLA & Self-Healing",
    "Cloud Utilization",
    "Action Log",
]:

    experiment = current_experiment()

    if experiment is None:
        st.warning("No new experiment is available. Upload a CSV and run it first.")
        if st.button("Return to Home"):
            navigate("Home")
            st.rerun()

    else:
        summary = experiment["summary"]
        observations = experiment["observations"]
        actions = experiment["actions"]
        capacity = experiment["capacity"]

        st.markdown(
            f"""
            <div class="hero" style="padding:1.2rem 1.5rem;">
                <div class="hero-chip">Experiment results</div>
                <div class="hero-title" style="font-size:2rem;">{st.session_state.page}</div>
                <div class="hero-copy" style="font-size:.92rem;">
                    Workload: {st.session_state.uploaded_filename}
                    &nbsp; · &nbsp; Run ID: {st.session_state.experiment_id}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # OVERVIEW
        # ----------------------------------------------------

        if st.session_state.page == "Overview":

            st.subheader("Policy comparison overview")

            risk = summary[summary["policy"] == "Risk-Aware"]
            reactive = summary[summary["policy"] == "Reactive"]
            static = summary[summary["policy"] == "Static"]

            if risk.empty:
                st.warning("Risk-Aware metrics are missing.")
            else:
                risk = risk.iloc[0]

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Availability", format_metric(risk["availability_percent"], "%"))
                m2.metric("SLA Violation Rate", format_metric(risk["sla_violation_rate_percent"], "%"))
                m3.metric("Migrations", format_metric(risk["migrations"], decimals=0))
                m4.metric("Mean Recovery", format_metric(risk["mean_recovery_time_minutes"], " min"))
                m5.metric(
                    "Overall Capacity",
                    format_metric(risk["mean_capacity_utilization_percent"], "%"),
                )

            left, right = st.columns(2)

            with left:
                fig = px.bar(
                    summary,
                    x="policy",
                    y="availability_percent",
                    color="policy",
                    text_auto=".2f",
                    title="Availability by Policy",
                )
                st.plotly_chart(chart_layout(fig), use_container_width=True)

            with right:
                fig = px.bar(
                    summary,
                    x="policy",
                    y="sla_violation_rate_percent",
                    color="policy",
                    text_auto=".2f",
                    title="SLA Violation Rate by Policy",
                )
                st.plotly_chart(chart_layout(fig), use_container_width=True)

            left, right = st.columns(2)

            with left:
                fig = px.bar(
                    summary,
                    x="policy",
                    y="migrations",
                    color="policy",
                    text_auto=True,
                    title="Migrations by Policy",
                )
                st.plotly_chart(chart_layout(fig), use_container_width=True)

            with right:
                fig = px.bar(
                    summary,
                    x="policy",
                    y="mean_recovery_time_minutes",
                    color="policy",
                    text_auto=".2f",
                    title="Mean Recovery Time",
                )
                st.plotly_chart(chart_layout(fig), use_container_width=True)

            st.subheader("Complete policy comparison")
            st.dataframe(summary, use_container_width=True, hide_index=True)

        # ----------------------------------------------------
        # WORKLOAD AND ML
        # ----------------------------------------------------

        elif st.session_state.page == "Workload & ML":

            st.subheader("Prepared workload")

            st.write(
                "The uploaded workload was validated and passed through the "
                "saved CPU prediction and anomaly detection models."
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("VMs", f"{observations['vm_id'].nunique():,}")
            m2.metric("Simulation Steps", f"{observations['step'].nunique():,}")
            m3.metric("Observation Rows", f"{len(observations):,}")

            vm_choice = st.selectbox(
                "Select a VM",
                sorted(observations["vm_id"].astype(str).unique()),
            )

            vm_data = observations[
                observations["vm_id"].astype(str) == vm_choice
            ].sort_values("step")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=vm_data["step"],
                y=vm_data["actual_cpu"],
                mode="lines",
                name="Actual CPU",
            ))
            fig.add_trace(go.Scatter(
                x=vm_data["step"],
                y=vm_data["predicted_cpu"],
                mode="lines",
                name="Predicted CPU",
            ))
            fig.update_layout(title=f"Actual vs Predicted CPU · {vm_choice}")
            st.plotly_chart(chart_layout(fig, 440), use_container_width=True)

            st.subheader("Anomaly detections")
            anomaly_rows = observations[observations["anomaly_flag"] == 1]
            st.metric("Flagged VM-steps", f"{len(anomaly_rows):,}")
            st.dataframe(
                anomaly_rows.head(500),
                use_container_width=True,
                hide_index=True,
            )

            download_df(
                observations,
                f"{st.session_state.experiment_id}_observations.csv",
                "Download Workload Observations",
            )

        # ----------------------------------------------------
        # ORCHESTRATION RESULTS
        # ----------------------------------------------------

        elif st.session_state.page == "Orchestration Results":

            st.subheader("Orchestration policy comparison")

            chart_specs = [
                ("availability_percent", "Availability (%)"),
                ("sla_violation_rate_percent", "SLA Violation Rate (%)"),
                ("migrations", "Migrations"),
                ("mean_recovery_time_minutes", "Mean Recovery (minutes)"),
                ("mean_capacity_utilization_percent", "Overall Capacity Utilization (%)"),
                ("illustrative_cost_units", "Illustrative Cost Units"),
            ]

            for index in range(0, len(chart_specs), 2):
                col1, col2 = st.columns(2)

                for col, spec in zip([col1, col2], chart_specs[index:index + 2]):
                    column, title = spec
                    with col:
                        if column in summary.columns:
                            fig = px.bar(
                                summary,
                                x="policy",
                                y=column,
                                color="policy",
                                text_auto=".2f",
                                title=title,
                            )
                            st.plotly_chart(chart_layout(fig), use_container_width=True)

            st.dataframe(summary, use_container_width=True, hide_index=True)

            download_df(
                summary,
                f"{st.session_state.experiment_id}_policy_comparison.csv",
                "Download Policy Comparison CSV",
            )

        # ----------------------------------------------------
        # SLA AND SELF-HEALING
        # ----------------------------------------------------

        elif st.session_state.page == "SLA & Self-Healing":

            st.subheader("SLA and simulated self-healing")

            left, right = st.columns(2)

            with left:
                fig = px.bar(
                    summary,
                    x="policy",
                    y="availability_percent",
                    color="policy",
                    text_auto=".2f",
                    title="Service Availability",
                )
                st.plotly_chart(chart_layout(fig), use_container_width=True)

            with right:
                fig = px.bar(
                    summary,
                    x="policy",
                    y="sla_violation_rate_percent",
                    color="policy",
                    text_auto=".2f",
                    title="SLA Violation Rate",
                )
                st.plotly_chart(chart_layout(fig), use_container_width=True)

            st.subheader("Availability over simulation time")

            selected_policy = st.selectbox(
                "Policy",
                ["Static", "Reactive", "Risk-Aware"],
                key="sla_policy",
            )

            selected = observations[observations["policy"] == selected_policy]

            availability_by_step = (
                selected.groupby("step")["service_available"]
                .mean()
                .mul(100)
                .reset_index(name="availability_percent")
            )

            fig = px.line(
                availability_by_step,
                x="step",
                y="availability_percent",
                title=f"{selected_policy} Availability Over Time",
            )
            st.plotly_chart(chart_layout(fig, 430), use_container_width=True)

            st.subheader("SLA violation breakdown")

            breakdown_cols = [
                column for column in [
                    "policy",
                    "availability_violation",
                    "latency_violation",
                    "cpu_violation",
                    "sla_violation",
                ]
                if column in observations.columns
            ]

            breakdown = (
                observations.groupby("policy")[breakdown_cols[1:]]
                .sum()
                .reset_index()
            )

            st.dataframe(breakdown, use_container_width=True, hide_index=True)

        # ----------------------------------------------------
        # CLOUD UTILIZATION
        # ----------------------------------------------------

        elif st.session_state.page == "Cloud Utilization":

            st.subheader("Cloud utilization")

            selected_policy = st.selectbox(
                "Select policy",
                ["Static", "Reactive", "Risk-Aware"],
                key="capacity_policy",
            )

            filtered = capacity[capacity["policy"] == selected_policy]

            if not filtered.empty:
                fig = px.line(
                    filtered,
                    x="step",
                    y="utilization_percent",
                    color="cloud",
                    title=f"{selected_policy} · Per-cloud utilization",
                )
                st.plotly_chart(chart_layout(fig, 450), use_container_width=True)

                if "platform_utilization_percent" in filtered.columns:
                    overall = (
                        filtered.groupby("step")["platform_utilization_percent"]
                        .first()
                        .reset_index()
                    )

                    fig = px.line(
                        overall,
                        x="step",
                        y="platform_utilization_percent",
                        title="Platform-wide utilization",
                    )
                    st.plotly_chart(chart_layout(fig, 380), use_container_width=True)

                st.caption(
                    "Platform-wide utilization = total CPU load ÷ combined cloud capacity. "
                    "Per-cloud utilization is shown separately."
                )

                st.dataframe(filtered, use_container_width=True, hide_index=True)

                download_df(
                    filtered,
                    f"{st.session_state.experiment_id}_{selected_policy}_capacity.csv",
                    "Download Capacity CSV",
                )

        # ----------------------------------------------------
        # ACTION LOG
        # ----------------------------------------------------

        elif st.session_state.page == "Action Log":

            st.subheader("Orchestration action log")

            if actions.empty:
                st.info(
                    "No orchestration actions were recorded for this experiment."
                )
            else:
                policy_filter = st.selectbox(
                    "Filter policy",
                    ["All Policies", "Static", "Reactive", "Risk-Aware"],
                )

                filtered_actions = actions.copy()

                if policy_filter != "All Policies":
                    filtered_actions = filtered_actions[
                        filtered_actions["policy"] == policy_filter
                    ]

                st.metric("Recorded actions", f"{len(filtered_actions):,}")
                st.dataframe(
                    filtered_actions,
                    use_container_width=True,
                    hide_index=True,
                )

                download_df(
                    filtered_actions,
                    f"{st.session_state.experiment_id}_action_log.csv",
                    "Download Action Log CSV",
                )

        # ----------------------------------------------------
        # COMMON DOWNLOAD AREA
        # ----------------------------------------------------

        st.divider()
        st.subheader("Download this experiment")

        d1, d2, d3, d4 = st.columns(4)

        with d1:
            download_df(
                summary,
                f"{st.session_state.experiment_id}_summary.csv",
                "Summary CSV",
            )

        with d2:
            download_df(
                observations,
                f"{st.session_state.experiment_id}_observations.csv",
                "Observations CSV",
            )

        with d3:
            download_df(
                actions,
                f"{st.session_state.experiment_id}_actions.csv",
                "Action Log CSV",
            )

        with d4:
            download_df(
                capacity,
                f"{st.session_state.experiment_id}_capacity.csv",
                "Capacity CSV",
            )

        st.caption(
            f"Experiment files are saved in: "
            f"{st.session_state.experiment_dir}"
        )

