from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# PROJECT PATHS AND IMPORTS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
RESULTS_DIR = ROOT / "outputs" / "experiment_results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prepare_simulation import prepare_uploaded_workload
from simulator import run_experiment


MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
VALIDATION_FILE = RESULTS_DIR / "validation_metrics.csv"
TEST_FILE = RESULTS_DIR / "test_metrics.csv"

EXPECTED_COLUMNS = [
    "vm_id",
    "timestamp",
    "cpu_percent",
    "memory_usage_kb",
    "disk_read_kbps",
    "disk_write_kbps",
    "network_received_kbps",
    "network_transmitted_kbps",
]

POLICIES = ["Static", "Reactive", "Risk-Aware"]

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aegis Cloud | Autonomous Orchestration",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #eaf1ff;
        --muted: #a8b5ce;
        --panel: #111c32;
        --panel2: #17243d;
        --line: rgba(160,184,230,.16);
        --blue: #6ea8fe;
        --cyan: #57e3d1;
        --purple: #a99bff;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(45,91,170,.20), transparent 32%),
            radial-gradient(circle at 90% 10%, rgba(60,190,180,.10), transparent 26%),
            #080f1d;
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #0d1728;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: var(--ink);
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.035em;
        color: #f2f6ff !important;
    }

    p, label, span, li {
        color: #c3cee2;
    }

    .hero {
        padding: 3.2rem 2.8rem;
        border: 1px solid rgba(130,170,255,.22);
        border-radius: 28px;
        background:
            linear-gradient(120deg, rgba(24,43,76,.96), rgba(13,25,45,.94)),
            #101b30;
        box-shadow: 0 24px 80px rgba(0,0,0,.25);
        position: relative;
        overflow: hidden;
        margin: .5rem 0 2rem 0;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 320px;
        height: 320px;
        right: -90px;
        top: -130px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(87,227,209,.17), transparent 70%);
        pointer-events: none;
    }

    .eyebrow {
        color: #78e7dc;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        color: #f4f7ff;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.1rem, 4vw, 3.7rem);
        font-weight: 800;
        line-height: 1.12;
        letter-spacing: -.055em;
        max-width: 850px;
        margin-bottom: 1rem;
    }

    .hero-copy {
        color: #b7c6df;
        font-size: 1.05rem;
        line-height: 1.8;
        max-width: 760px;
    }

    .hero-chip {
        display: inline-block;
        padding: .45rem .8rem;
        border-radius: 100px;
        border: 1px solid rgba(120,231,220,.3);
        color: #9af1e8;
        background: rgba(87,227,209,.08);
        font-size: .78rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
    }

    .section-kicker {
        color: #7ce5db;
        font-size: .76rem;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: .14em;
        margin-bottom: .45rem;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -.04em;
        color: #f2f6ff;
        margin-bottom: .4rem;
    }

    .section-copy {
        color: #aab8d0;
        margin-bottom: 1.25rem;
        line-height: 1.7;
    }

    .feature-card {
        background: linear-gradient(145deg, rgba(23,36,61,.96), rgba(14,25,43,.96));
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1.35rem;
        min-height: 210px;
        transition: all .2s ease;
    }

    .feature-card:hover {
        border-color: rgba(110,168,254,.48);
        transform: translateY(-3px);
        box-shadow: 0 14px 35px rgba(0,0,0,.18);
    }

    .card-icon {
        font-size: 1.65rem;
        margin-bottom: .7rem;
    }

    .card-title {
        color: #edf3ff;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: .55rem;
    }

    .card-copy {
        color: #aab8d0;
        font-size: .9rem;
        line-height: 1.65;
    }

    .step-card {
        background: rgba(18,30,51,.8);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem;
        height: 100%;
    }

    .step-number {
        color: #76e5db;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .1em;
    }

    .step-title {
        color: #edf3ff;
        font-weight: 800;
        margin: .45rem 0;
    }

    .step-copy {
        color: #aab8d0;
        font-size: .85rem;
        line-height: 1.6;
    }

    .status-pill {
        display: inline-block;
        background: rgba(87,227,209,.09);
        border: 1px solid rgba(87,227,209,.25);
        color: #91eee4;
        border-radius: 100px;
        padding: .4rem .75rem;
        font-size: .78rem;
        font-weight: 700;
    }

    .footer-note {
        color: #7e8da8;
        font-size: .8rem;
        text-align: center;
        padding: 1.5rem 0 .5rem 0;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #14223a, #101b2e);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #aab8d0;
    }

    div[data-testid="stMetricValue"] {
        color: #f2f6ff;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 2.8rem;
        font-weight: 700;
        border: 1px solid rgba(110,168,254,.25);
        transition: all .15s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #477fe0, #4e9be8);
        color: white;
        border: none;
    }

    .stButton > button:hover {
        border-color: #78e7dc;
        transform: translateY(-1px);
    }

    [data-testid="stFileUploader"] {
        background: rgba(10,19,34,.45);
        border: 1px dashed rgba(110,168,254,.38);
        border-radius: 14px;
        padding: .5rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: var(--line) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    path = Path(path)
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def fmt(value, suffix="", decimals=2):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}{suffix}"


def metric_value(data, column, policy="Risk-Aware"):
    if data is None or data.empty or column not in data.columns:
        return None

    if "policy" in data.columns:
        selected = data[data["policy"].astype(str) == policy]
        if selected.empty:
            return None
        value = selected.iloc[0][column]
    else:
        value = data.iloc[0][column]

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def csv_download(data, filename, label):
    if data is not None and not data.empty:
        st.download_button(
            label=label,
            data=data.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
        )


def plot_bar(data, x, y, title, color=None):
    if data is None or data.empty or x not in data.columns or y not in data.columns:
        st.info("No data available for this chart yet.")
        return

    fig = px.bar(
        data,
        x=x,
        y=y,
        color=color,
        title=title,
        text_auto=".2f" if pd.api.types.is_numeric_dtype(data[y]) else True,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dbe7fb",
        title_font_color="#f2f6ff",
        height=380,
        margin=dict(l=15, r=15, t=55, b=15),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_line(data, x, y, title, color=None):
    if data is None or data.empty or x not in data.columns or y not in data.columns:
        st.info("No data available for this chart yet.")
        return

    fig = px.line(data, x=x, y=y, color=color, title=title)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dbe7fb",
        title_font_color="#f2f6ff",
        height=420,
        margin=dict(l=15, r=15, t=55, b=15),
    )
    st.plotly_chart(fig, use_container_width=True)


def make_sample_csv():
    rows = []

    for vm_number in range(1, 31):
        vm_id = f"sample_vm_{vm_number:02d}"

        for step in range(288):
            timestamp = pd.Timestamp("2025-01-01 00:00:00") + pd.Timedelta(
                minutes=5 * step
            )

            cpu = 25 + 15 * np.sin(step / 12) + (vm_number % 5) * 3
            cpu = float(np.clip(cpu, 0, 100))

            rows.append({
                "vm_id": vm_id,
                "timestamp": timestamp.isoformat(),
                "cpu_percent": round(cpu, 2),
                "memory_usage_kb": 500000 + vm_number * 1000,
                "disk_read_kbps": 100 + (step % 10) * 5,
                "disk_write_kbps": 80 + (step % 8) * 4,
                "network_received_kbps": 200 + (step % 12) * 10,
                "network_transmitted_kbps": 150 + (step % 9) * 8,
            })

    return pd.DataFrame(rows)


def validate_uploaded_csv(df):
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            "The uploaded CSV is missing these required columns: "
            + ", ".join(missing)
        )

    df = df[EXPECTED_COLUMNS].copy()

    if df.empty:
        raise ValueError("The uploaded CSV contains no data rows.")

    df["vm_id"] = df["vm_id"].astype(str).str.strip()

    if df["vm_id"].eq("").any():
        raise ValueError("Some rows have an empty vm_id.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    if df["timestamp"].isna().any():
        raise ValueError("Some timestamp values could not be parsed.")

    numeric_columns = EXPECTED_COLUMNS[2:]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_columns].isna().any().any():
        raise ValueError("Some numeric workload fields are empty or invalid.")

    if not np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("The CSV contains infinite or invalid numeric values.")

    if ((df["cpu_percent"] < 0) | (df["cpu_percent"] > 100)).any():
        raise ValueError("CPU utilization must be between 0 and 100.")

    if df.duplicated(["vm_id", "timestamp"]).any():
        raise ValueError("Duplicate VM and timestamp rows were found.")

    vm_count = df["vm_id"].nunique()
    rows_per_vm = df.groupby("vm_id").size()

    if vm_count != 30:
        raise ValueError(f"Expected 30 VMs, but found {vm_count}.")

    if not (rows_per_vm == 288).all():
        raise ValueError(
            "Each VM must contain exactly 288 workload records. "
            "That represents 24 hours at 5-minute intervals."
        )

    return df.sort_values(["vm_id", "timestamp"]).reset_index(drop=True)


def run_uploaded_experiment(uploaded_file):
    if uploaded_file is None:
        st.error("Upload a workload CSV before running the experiment.")
        return

    try:
        with st.spinner("Validating workload, running ML models, and simulating policies..."):
            raw_df = pd.read_csv(uploaded_file)
            clean_df = validate_uploaded_csv(raw_df)

            # prepare_uploaded_workload expects a DataFrame.
            workload, failures = prepare_uploaded_workload(
                upload_df=clean_df,
                num_vms=30,
                num_steps=288,
            )

            summary, observations, actions, capacity = run_experiment(
                workload=workload,
                failures=failures,
                output_dir=None,
            )

            st.session_state.experiment_results = {
                "summary": summary.copy(),
                "observations": observations.copy(),
                "actions": actions.copy(),
                "capacity": capacity.copy(),
            }
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.active_page = "Overview"

        st.success("Experiment completed. Your new results are ready.")

    except Exception as error:
        st.error(f"Experiment could not be completed: {error}")


# ============================================================
# ORIGINAL MODEL EVALUATION FILES
# These are shown only inside the results area, not on Home.
# ============================================================

model_metrics = load_csv(MODEL_METRICS_FILE)
validation_metrics = load_csv(VALIDATION_FILE)
test_metrics = load_csv(TEST_FILE)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

has_results = st.session_state.experiment_results is not None

st.sidebar.markdown("## ☁️ Aegis Cloud")
st.sidebar.caption("Autonomous Multi-Cloud Orchestration")
st.sidebar.divider()

if has_results:
    nav_options = [
        "Home",
        "Overview",
        "Workload & ML",
        "Orchestration Results",
        "SLA & Self-Healing",
        "Cloud Utilization",
        "Action Log",
        "About",
    ]
else:
    nav_options = ["Home", "About"]

page = st.sidebar.radio(
    "Navigate",
    nav_options,
    index=nav_options.index(st.session_state.active_page)
    if st.session_state.active_page in nav_options
    else 0,
)

st.session_state.active_page = page

st.sidebar.divider()
st.sidebar.markdown("### Experiment Workspace")
st.sidebar.caption("Upload a standardized workload CSV to begin.")

sidebar_upload = st.sidebar.file_uploader(
    "Workload CSV",
    type=["csv"],
    key="sidebar_workload_upload",
    help="30 VMs × 288 five-minute records per VM.",
)

if st.sidebar.button(
    "▶ Run Experiment",
    type="primary",
    use_container_width=True,
):
    run_uploaded_experiment(sidebar_upload)
    st.rerun()

sample_df = make_sample_csv()

st.sidebar.download_button(
    "⬇ Download Sample CSV",
    data=sample_df.to_csv(index=False).encode("utf-8"),
    file_name="sample_standardized_workload.csv",
    mime="text/csv",
    use_container_width=True,
)

if has_results:
    st.sidebar.success("Experiment results are available.")
    st.sidebar.caption(f"Workload: {st.session_state.uploaded_filename}")
    if st.sidebar.button("Clear Current Results", use_container_width=True):
        st.session_state.experiment_results = None
        st.session_state.uploaded_filename = None
        st.session_state.active_page = "Home"
        st.rerun()

st.sidebar.caption(
    "This platform simulates orchestration and recovery. "
    "It does not provision or migrate live cloud VMs."
)

# ============================================================
# HOME / LANDING PAGE
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-chip">ML-assisted infrastructure intelligence</div>
            <div class="eyebrow">Autonomous cloud systems · Research prototype</div>
            <div class="hero-title">
                Orchestrate smarter.<br>
                Recover faster.
            </div>
            <div class="hero-copy">
                An interactive platform for exploring workload prediction,
                anomaly detection, multi-cloud resource orchestration, and
                simulated self-healing under infrastructure failures.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">Start here</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Explore the platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Choose a starting point. Results are generated only after you upload a workload and run a new experiment.</div>',
        unsafe_allow_html=True,
    )

    card1, card2, card3 = st.columns(3, gap="large")

    with card1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-icon">📂</div>
                <div class="card-title">Upload Workload</div>
                <div class="card-copy">
                    Bring your standardized workload CSV into the experiment
                    workspace. Validate VM telemetry and prepare it for ML analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        home_upload = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            key="home_workload_upload",
            label_visibility="collapsed",
        )

        if home_upload:
            st.caption(f"Selected: **{home_upload.name}**")

        if st.button(
            "Run New Experiment",
            type="primary",
            use_container_width=True,
            disabled=home_upload is None,
        ):
            run_uploaded_experiment(home_upload)
            st.rerun()

        st.download_button(
            "Download Sample Workload",
            data=sample_df.to_csv(index=False).encode("utf-8"),
            file_name="sample_standardized_workload.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with card2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-icon">🧭</div>
                <div class="card-title">About the Platform</div>
                <div class="card-copy">
                    Understand the research objective, architecture,
                    orchestration strategies, technologies, and experiment scope.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Explore the Project", use_container_width=True):
            st.session_state.active_page = "About"
            st.rerun()

    with card3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="card-icon">⚙️</div>
                <div class="card-title">How It Works</div>
                <div class="card-copy">
                    Follow the pipeline from workload ingestion to ML signals,
                    policy decisions, simulated recovery, and evaluation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("View Workflow", use_container_width=True):
            st.session_state.active_page = "About"
            st.rerun()

    st.markdown("---")

    st.markdown('<div class="section-kicker">Platform capabilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Infrastructure intelligence in one workspace</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3, gap="large")

    with f1:
        st.markdown(
            """
            <div class="step-card">
                <div class="card-title">📈 CPU Prediction</div>
                <div class="card-copy">
                    Random Forest predicts next-step CPU utilization from
                    historical workload features.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f2:
        st.markdown(
            """
            <div class="step-card">
                <div class="card-title">🔎 Anomaly Detection</div>
                <div class="card-copy">
                    Isolation Forest identifies unusual workload behavior
                    that may indicate elevated operational risk.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f3:
        st.markdown(
            """
            <div class="step-card">
                <div class="card-title">🛡️ Self-Healing Simulation</div>
                <div class="card-copy">
                    Compare Static, Reactive, and Risk-Aware policies under
                    simulated cloud failures and capacity pressure.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown('<div class="section-kicker">Experiment pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">From telemetry to decisions</div>', unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns(5, gap="small")

    steps = [
        ("01", "Upload", "Standardized VM workload CSV"),
        ("02", "Predict", "RF CPU forecast + IF anomaly score"),
        ("03", "Orchestrate", "Evaluate three placement policies"),
        ("04", "Recover", "Simulate failure response and migration"),
        ("05", "Evaluate", "Availability, SLA, utilization, cost"),
    ]

    for col, step in zip([s1, s2, s3, s4, s5], steps):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-number">{step[0]}</div>
                    <div class="step-title">{step[1]}</div>
                    <div class="step-copy">{step[2]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="footer-note">
            Aegis Cloud · Academic research prototype · Simulated multi-cloud orchestration
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# RESULTS PAGES
# ============================================================

elif page in [
    "Overview",
    "Workload & ML",
    "Orchestration Results",
    "SLA & Self-Healing",
    "Cloud Utilization",
    "Action Log",
]:

    if not has_results:
        st.warning("No experiment results are available. Return to Home and upload a workload.")
        st.stop()

    active = st.session_state.experiment_results
    summary = active["summary"]
    observations = active["observations"]
    action_log = active["actions"]
    capacity = active["capacity"]

    st.markdown(
        """
        <div class="eyebrow">Experiment workspace</div>
        <div class="section-heading">Your orchestration results</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="status-pill">Fresh experiment · {st.session_state.uploaded_filename}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    if page == "Overview":
        st.subheader("Experiment Overview")

        availability = metric_value(summary, "availability_percent")
        sla_rate = metric_value(summary, "sla_violation_rate_percent")
        migrations = metric_value(summary, "migrations")
        recovery = metric_value(summary, "mean_recovery_time_minutes")
        utilization = metric_value(summary, "mean_capacity_utilization_percent")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Availability", fmt(availability, "%"))
        c2.metric("SLA Violation Rate", fmt(sla_rate, "%"))
        c3.metric("Migrations", fmt(migrations, decimals=0))
        c4.metric("Mean Recovery", fmt(recovery, " min"))
        c5.metric("Capacity Utilization", fmt(utilization, "%"))

        st.divider()

        left, right = st.columns(2)

        with left:
            plot_bar(summary, "policy", "availability_percent", "Availability by Policy")

        with right:
            plot_bar(summary, "policy", "sla_violation_rate_percent", "SLA Violation Rate by Policy")

        left, right = st.columns(2)

        with left:
            if not capacity.empty and "Risk-Aware" in capacity.get("policy", pd.Series()).astype(str).values:
                risk_capacity = capacity[capacity["policy"] == "Risk-Aware"]
                plot_line(
                    risk_capacity,
                    "step",
                    "utilization_percent",
                    "Risk-Aware Cloud Capacity Over Time",
                    color="cloud",
                )
            else:
                st.info("Capacity timeline is not available.")

        with right:
            st.subheader("Policy Comparison")
            st.dataframe(summary, use_container_width=True, hide_index=True)

        st.subheader("Key Insights")

        static_availability = metric_value(summary, "availability_percent", "Static")
        risk_availability = metric_value(summary, "availability_percent", "Risk-Aware")
        static_sla = metric_value(summary, "sla_violation_rate_percent", "Static")
        risk_sla = metric_value(summary, "sla_violation_rate_percent", "Risk-Aware")

        if static_availability is not None and risk_availability is not None:
            st.info(
                f"Risk-Aware availability changed by "
                f"{risk_availability - static_availability:+.2f} percentage points "
                f"compared with Static."
            )

        if static_sla is not None and risk_sla is not None:
            if static_sla > 0:
                reduction = (static_sla - risk_sla) / static_sla * 100
                st.success(f"Risk-Aware SLA violation rate changed by {reduction:.1f}% relative to Static.")

        csv_download(summary, "orchestration_policy_comparison.csv", "Download Policy Summary")

    # --------------------------------------------------------
    # WORKLOAD & ML
    # --------------------------------------------------------

    elif page == "Workload & ML":
        st.subheader("Workload and Machine Learning")

        st.markdown(
            """
            The ML stage uses the saved models to estimate CPU utilization
            and detect unusual workload behavior. Model evaluation tables
            below are the existing offline evaluation results from the project.
            """
        )

        st.markdown("### Model Summary")

        m1, m2 = st.columns(2)

        with m1:
            st.markdown(
                """
                **Random Forest Regressor**
                - Predicts next-step CPU utilization
                - Uses workload history and resource telemetry
                - Supports risk-aware orchestration decisions
                """
            )

        with m2:
            st.markdown(
                """
                **Isolation Forest**
                - Detects unusual workload patterns
                - Produces anomaly signals for the policy
                - Helps identify potentially risky VM behavior
                """
            )

        st.markdown("### Test Metrics")
        if not test_metrics.empty:
            st.dataframe(test_metrics, use_container_width=True, hide_index=True)
        else:
            st.info("Test metrics file not found.")

        st.markdown("### Validation Metrics")
        if not validation_metrics.empty:
            st.dataframe(validation_metrics, use_container_width=True, hide_index=True)
        else:
            st.info("Validation metrics file not found.")

        st.markdown("### Training Metrics")
        if not model_metrics.empty:
            st.dataframe(model_metrics, use_container_width=True, hide_index=True)
        else:
            st.info("Training metrics file not found.")

    # --------------------------------------------------------
    # ORCHESTRATION RESULTS
    # --------------------------------------------------------

    elif page == "Orchestration Results":
        st.subheader("Orchestration Policy Comparison")
        st.write("Compare Static, Reactive, and Risk-Aware orchestration strategies.")

        charts = [
            ("availability_percent", "Availability by Policy"),
            ("sla_violation_rate_percent", "SLA Violation Rate by Policy"),
            ("migrations", "Migration Count by Policy"),
            ("mean_recovery_time_minutes", "Mean Recovery Time by Policy"),
            ("mean_capacity_utilization_percent", "Mean Capacity Utilization"),
            ("illustrative_cost_units", "Illustrative Cost Comparison"),
        ]

        for i in range(0, len(charts), 2):
            left, right = st.columns(2)
            with left:
                plot_bar(summary, "policy", charts[i][0], charts[i][1])
            if i + 1 < len(charts):
                with right:
                    plot_bar(summary, "policy", charts[i + 1][0], charts[i + 1][1])

        st.subheader("Full Policy Summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        csv_download(summary, "orchestration_policy_comparison.csv", "Download Policy Comparison CSV")

    # --------------------------------------------------------
    # SLA & SELF-HEALING
    # --------------------------------------------------------

    elif page == "SLA & Self-Healing":
        st.subheader("SLA and Self-Healing Analysis")

        left, right = st.columns(2)

        with left:
            plot_bar(summary, "policy", "sla_violation_rate_percent", "SLA Violation Rate")

        with right:
            plot_bar(summary, "policy", "mean_recovery_time_minutes", "Mean Recovery Time")

        if "performance_sla_violations" in summary.columns:
            plot_bar(
                summary,
                "policy",
                "performance_sla_violations",
                "Performance SLA Violations",
            )

        display_columns = [
            c for c in [
                "policy",
                "availability_percent",
                "sla_violation_rate_percent",
                "unavailable_vm_steps",
                "performance_sla_violations",
                "migrations",
                "mean_recovery_time_minutes",
            ]
            if c in summary.columns
        ]

        st.subheader("Recovery Metrics")
        st.dataframe(summary[display_columns], use_container_width=True, hide_index=True)

        st.info(
            "SLA violation rate includes unavailable VM-steps. "
            "CPU and latency violations are tracked separately. "
            "Recovery and migration behavior are simulated."
        )

    # --------------------------------------------------------
    # CLOUD UTILIZATION
    # --------------------------------------------------------

    elif page == "Cloud Utilization":
        st.subheader("Cloud Capacity Utilization")

        if capacity.empty:
            st.warning("Capacity data is not available.")
        else:
            policies = capacity["policy"].dropna().astype(str).unique().tolist()

            default_index = policies.index("Risk-Aware") if "Risk-Aware" in policies else 0

            selected_policy = st.selectbox(
                "Select orchestration policy",
                policies,
                index=default_index,
            )

            filtered = capacity[capacity["policy"].astype(str) == selected_policy]

            plot_line(
                filtered,
                "step",
                "utilization_percent",
                f"{selected_policy} Cloud Utilization Over Time",
                color="cloud",
            )

            st.subheader("Capacity Data")
            st.dataframe(filtered, use_container_width=True, hide_index=True)

            csv_download(
                filtered,
                f"capacity_utilization_{selected_policy}.csv",
                "Download Capacity CSV",
            )

    # --------------------------------------------------------
    # ACTION LOG
    # --------------------------------------------------------

    elif page == "Action Log":
        st.subheader("Orchestration Action Log")
        st.write("Review simulated orchestration actions and migration decisions.")

        if action_log.empty:
            st.info("No action log entries were generated.")
        else:
            if "policy" in action_log.columns:
                policy_options = ["All Policies"] + (
                    action_log["policy"].dropna().astype(str).unique().tolist()
                )

                selected_policy = st.selectbox("Filter by policy", policy_options)

                if selected_policy == "All Policies":
                    filtered_log = action_log
                else:
                    filtered_log = action_log[
                        action_log["policy"].astype(str) == selected_policy
                    ]
            else:
                filtered_log = action_log

            st.dataframe(filtered_log, use_container_width=True, hide_index=True)
            csv_download(filtered_log, "orchestration_action_log.csv", "Download Action Log")

# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":
    st.markdown('<div class="eyebrow">Project information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">About Aegis Cloud</div>', unsafe_allow_html=True)

    st.markdown(
        """
        Aegis Cloud is an academic research prototype that explores
        **machine-learning-assisted multi-cloud resource orchestration**
        and simulated self-healing infrastructure behavior.

        ### Research objective
        Compare orchestration policies and evaluate how workload prediction,
        anomaly signals, and failure-aware decisions affect service continuity.

        ### Core components
        1. Workload preprocessing and validation
        2. Random Forest CPU prediction
        3. Isolation Forest anomaly detection
        4. Multi-cloud orchestration simulator
        5. Simulated failure detection and VM migration
        6. SLA, recovery, utilization, and cost analysis
        7. Interactive Streamlit dashboard

        ### Orchestration policies
        - **Static:** retains the initial VM placement.
        - **Reactive:** responds to cloud failures and CPU overload.
        - **Risk-Aware:** considers failure signals, risk, anomaly detection,
          CPU behavior, latency, and projected capacity.

        ### Technologies
        Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, Matplotlib,
        and Joblib.

        ### Experiment scope
        The platform uses workload data and saved ML models, but the
        multi-cloud environment, failure response, recovery, and migration
        are simulated. It does not provision or migrate live VMs between
        cloud providers.
        """
    )

    st.markdown("---")
    st.subheader("How the platform works")

    a, b = st.columns(2)

    with a:
        st.markdown(
            """
            **01 · Workload ingestion**

            Upload a standardized CSV containing VM telemetry at five-minute
            intervals. The platform checks the structure and required fields.

            **02 · ML analysis**

            The saved Random Forest model predicts CPU utilization, while
            Isolation Forest detects unusual workload patterns.

            **03 · Policy evaluation**

            The simulator runs Static, Reactive, and Risk-Aware policies
            against the same workload and simulated failure schedule.
            """
        )

    with b:
        st.markdown(
            """
            **04 · Simulated self-healing**

            The orchestration logic evaluates failures and resource pressure,
            then simulates recovery and VM migration actions.

            **05 · Results**

            The dashboard compares availability, SLA violations, migrations,
            recovery time, capacity utilization, and illustrative cost.

            **06 · Export**

            Download experiment summaries, capacity records, and action logs
            as CSV files for further analysis.
            """
        )

    st.markdown("---")
    st.caption(
        "Research prototype · Simulation-based evaluation · "
        "No live cloud resources are provisioned or migrated"
    )