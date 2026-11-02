from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT / "src"
RESULTS_DIR = ROOT / "outputs" / "experiment_results"
GRAPH_DIR = RESULTS_DIR / "graphs"

SIMULATOR_FILE = SRC_DIR / "simulator.py"
PLOT_FILE = SRC_DIR / "plot_results.py"

SUMMARY_FILE = RESULTS_DIR / "orchestration_policy_comparison.csv"
OBSERVATIONS_FILE = RESULTS_DIR / "orchestration_step_observations.csv.gz"
ACTION_LOG_FILE = RESULTS_DIR / "orchestration_action_log.csv"
CAPACITY_FILE = RESULTS_DIR / "cloud_capacity_utilization.csv"

MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
VALIDATION_FILE = RESULTS_DIR / "validation_metrics.csv"
TEST_FILE = RESULTS_DIR / "test_metrics.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Autonomous Multi-Cloud Orchestration",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS — LIGHT, SIMPLE, PROFESSIONAL
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #172554;
        margin-bottom: 0;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-top: 0.2rem;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #172554;
        margin-top: 0.5rem;
    }

    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #dbeafe;
        padding: 14px;
        border-radius: 12px;
    }

    .status-box {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        padding: 12px;
        border-radius: 10px;
        color: #065f46;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_observations(path):
    if path.exists():
        return pd.read_csv(path, compression="gzip")
    return pd.DataFrame()


def clear_data_cache():
    load_csv.clear()
    load_observations.clear()


summary = load_csv(SUMMARY_FILE)
model_metrics = load_csv(MODEL_METRICS_FILE)
validation_metrics = load_csv(VALIDATION_FILE)
test_metrics = load_csv(TEST_FILE)
capacity = load_csv(CAPACITY_FILE)
action_log = load_csv(ACTION_LOG_FILE)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_value(data, column, policy="Risk-Aware"):
    """Get a numeric value from the policy summary."""

    if data.empty or column not in data.columns:
        return None

    if "policy" in data.columns:
        selected = data[data["policy"].astype(str) == policy]

        if not selected.empty:
            value = selected.iloc[0][column]
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

    try:
        return float(data.iloc[0][column])
    except (TypeError, ValueError):
        return None


def format_value(value, suffix="", decimals=2):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}{suffix}"


def show_csv_download(data, filename, label):
    if not data.empty:
        st.download_button(
            label=label,
            data=data.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
        )


def show_graph(filename, caption=None):
    path = GRAPH_DIR / filename

    if path.exists():
        st.image(str(path), use_container_width=True)

        if caption:
            st.caption(caption)
    else:
        st.info(
            f"Graph not found: {filename}. "
            "Run the experiment to generate the graphs."
        )


def run_script(script_path, label):
    if not script_path.exists():
        st.error(f"Cannot find script: {script_path}")
        return False

    with st.spinner(f"{label} is running... Please wait."):
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )

    if result.returncode == 0:
        st.success(f"{label} completed successfully.")

        if result.stdout:
            with st.expander(f"{label} output"):
                st.code(result.stdout)

        return True

    st.error(f"{label} failed.")

    if result.stdout:
        st.code(result.stdout)

    if result.stderr:
        st.code(result.stderr)

    return False


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("☁️ Multi-Cloud")
st.sidebar.subheader("Orchestration Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Workload & ML",
        "Orchestration Results",
        "SLA & Self-Healing",
        "Cloud Utilization",
        "Action Log",
        "About",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Experiment")

st.sidebar.write(
    "Run the orchestration simulator and regenerate the result graphs."
)

if st.sidebar.button(
    "▶ Run Experiment",
    type="primary",
    use_container_width=True,
):
    success = run_script(
        SIMULATOR_FILE,
        "Orchestration experiment",
    )

    if success:
        clear_data_cache()

        plot_success = run_script(
            PLOT_FILE,
            "Result graph generation",
        )

        if plot_success:
            clear_data_cache()
            st.rerun()


st.sidebar.caption(
    "Experiment outputs are saved in "
    "outputs/experiment_results/"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Autonomous Multi-Cloud Orchestration'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'A Self-Healing, Risk-Aware Framework for Cloud Service Continuity using ML'
    '</div>',
    unsafe_allow_html=True,
)

if not summary.empty:
    st.markdown(
        '<div class="status-box">✓ Experiment results loaded</div>',
        unsafe_allow_html=True,
    )
else:
    st.warning(
        "Experiment results have not been found yet. "
        "Use **Run Experiment** in the sidebar."
    )


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.subheader("Experiment Overview")

    availability = find_value(summary, "availability_percent")
    sla_rate = find_value(summary, "sla_violation_rate_percent")
    migrations = find_value(summary, "migrations")
    recovery = find_value(summary, "mean_recovery_time_minutes")
    utilization = find_value(
        summary,
        "mean_capacity_utilization_percent",
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Availability",
        format_value(availability, "%"),
        "Risk-Aware",
    )

    col2.metric(
        "SLA Violation Rate",
        format_value(sla_rate, "%"),
        "Risk-Aware",
    )

    col3.metric(
        "Total Migrations",
        format_value(migrations, decimals=0),
        "Risk-Aware",
    )

    col4.metric(
        "Mean Recovery Time",
        format_value(recovery, " min"),
        "Risk-Aware",
    )

    col5.metric(
        "Mean Capacity Utilization",
        format_value(utilization, "%"),
        "Risk-Aware",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Availability Comparison")
        show_graph("01_availability_comparison.png")

    with right:
        st.subheader("SLA Violation Rate")
        show_graph("02_sla_violation_comparison.png")

    left, right = st.columns(2)

    with left:
        st.subheader("Cloud Capacity Over Time")
        show_graph("07_capacity_over_time_risk_aware.png")

    with right:
        st.subheader("Policy Comparison Summary")

        if not summary.empty:
            display_columns = [
                column
                for column in [
                    "policy",
                    "availability_percent",
                    "sla_violation_rate_percent",
                    "migrations",
                    "mean_recovery_time_minutes",
                ]
                if column in summary.columns
            ]

            st.dataframe(
                summary[display_columns],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Run the experiment to display the policy summary.")

    st.subheader("Key Insights")

    if not summary.empty:

        risk_availability = find_value(
            summary,
            "availability_percent",
            "Risk-Aware",
        )

        static_sla = find_value(
            summary,
            "sla_violation_rate_percent",
            "Static",
        )

        risk_sla = find_value(
            summary,
            "sla_violation_rate_percent",
            "Risk-Aware",
        )

        if risk_availability is not None:
            st.success(
                f"Risk-Aware availability: {risk_availability:.2f}%"
            )

        if (
            static_sla is not None
            and risk_sla is not None
            and static_sla > 0
        ):
            reduction = (
                (static_sla - risk_sla)
                / static_sla
                * 100
            )

            st.success(
                f"SLA violation rate reduced by "
                f"{reduction:.1f}% compared with Static."
            )

        st.info(
            "The experiment compares Static, Reactive, and Risk-Aware "
            "orchestration policies using the same initial VM placement."
        )


# ============================================================
# WORKLOAD & ML PAGE
# ============================================================

elif page == "Workload & ML":

    st.subheader("Workload and Machine Learning")

    st.write(
        "The machine learning component predicts the next CPU utilization "
        "value and detects anomalous workload behavior."
    )

    st.markdown("### Model Evaluation")

    if not test_metrics.empty:
        st.write("Final test results on held-out test VMs:")

        st.dataframe(
            test_metrics,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Test metrics file not found.")

    if not validation_metrics.empty:
        st.write("Validation results:")

        st.dataframe(
            validation_metrics,
            use_container_width=True,
            hide_index=True,
        )

    if not model_metrics.empty:
        st.write("Model training metrics:")

        st.dataframe(
            model_metrics,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Model Summary")

    st.markdown(
        """
        - **Random Forest:** predicts next-step CPU utilization.
        - **Persistence baseline:** uses the current CPU value as the next prediction.
        - **Isolation Forest:** detects unusual workload behavior.
        - **VM-level split:** separates train, validation, and test VMs to avoid VM leakage.
        """
    )

    st.markdown("### Test Metrics")

    if not test_metrics.empty:
        st.caption(
            "Use the table above as the source of truth for final test metrics."
        )


# ============================================================
# ORCHESTRATION RESULTS PAGE
# ============================================================

elif page == "Orchestration Results":

    st.subheader("Orchestration Policy Comparison")

    st.write(
        "Compare the Static, Reactive, and Risk-Aware policies."
    )

    left, right = st.columns(2)

    with left:
        show_graph("01_availability_comparison.png")

    with right:
        show_graph("02_sla_violation_comparison.png")

    left, right = st.columns(2)

    with left:
        show_graph("03_migration_comparison.png")

    with right:
        show_graph("04_recovery_time_comparison.png")

    left, right = st.columns(2)

    with left:
        show_graph("05_capacity_utilization_comparison.png")

    with right:
        show_graph("06_illustrative_cost_comparison.png")

    st.subheader("Policy Summary Table")

    if not summary.empty:
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

        show_csv_download(
            summary,
            "orchestration_policy_comparison.csv",
            "Download Policy Comparison CSV",
        )


# ============================================================
# SLA & SELF-HEALING PAGE
# ============================================================

elif page == "SLA & Self-Healing":

    st.subheader("SLA and Self-Healing Results")

    st.write(
        "This section presents service availability, SLA violations, "
        "outages, migration actions, and recovery behavior."
    )

    left, right = st.columns(2)

    with left:
        show_graph("02_sla_violation_comparison.png")

    with right:
        show_graph("08_sla_components_comparison.png")

    st.subheader("Recovery Time")
    show_graph("04_recovery_time_comparison.png")

    st.subheader("Policy Metrics")

    if not summary.empty:

        columns = [
            column
            for column in [
                "policy",
                "availability_percent",
                "sla_violation_rate_percent",
                "unavailable_vm_steps",
                "performance_sla_violations",
                "migrations",
                "mean_recovery_time_minutes",
            ]
            if column in summary.columns
        ]

        metrics_display = summary[columns].copy()

        metrics_display = metrics_display.rename(
            columns={
                "policy": "Orchestration Policy",
                "availability_percent": "Service Availability (%)",
                "sla_violation_rate_percent": "Overall SLA Violations (%)",
                "unavailable_vm_steps": "Unavailable VM-Steps",
                "performance_sla_violations": (
                    "Performance Violations (CPU + Latency)"
                ),
                "migrations": "Migrations",
                "mean_recovery_time_minutes": "Mean Recovery Time (min)",
            }
        )

        st.dataframe(
            metrics_display,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### How to Interpret These Results")

        st.markdown(
            """
            **Overall SLA violations** include unavailable VM-steps and
            performance violations.

            **Performance violations** count CPU or latency threshold
            breaches while the VM is available.

            In this experiment, performance violations are **0** because
            no available VM-step crossed the configured CPU threshold of
            **85%** or latency threshold of **100 ms**.

            The non-zero overall SLA violation rate is caused by service
            unavailability during cloud failures and migration interruptions.
            """
        )

        st.success(
            "Risk-Aware orchestration achieved the highest service availability "
            "and the lowest overall SLA violation rate in this experiment."
        )

    else:
        st.warning("Run the experiment to display SLA metrics.")


# ============================================================
# CLOUD UTILIZATION PAGE
# ============================================================

elif page == "Cloud Utilization":

    st.subheader("Cloud Capacity Utilization")

    if capacity.empty:
        st.warning(
            "Capacity data not found. Run the experiment first."
        )

    else:

        policies = (
            capacity["policy"].dropna().astype(str).unique().tolist()
            if "policy" in capacity.columns
            else []
        )

        if not policies:
            st.warning("No orchestration policy data is available.")

        else:

            selected_policy = st.selectbox(
                "Select orchestration policy",
                policies,
                index=(
                    policies.index("Risk-Aware")
                    if "Risk-Aware" in policies
                    else 0
                ),
            )

            graph_name = (
                "07_capacity_over_time_"
                f"{selected_policy.lower().replace('-', '_')}.png"
            )

            show_graph(graph_name)

            filtered_capacity = capacity[
                capacity["policy"].astype(str) == selected_policy
            ]

            st.subheader("Capacity Data")

            st.dataframe(
                filtered_capacity,
                use_container_width=True,
                hide_index=True,
            )

            show_csv_download(
                filtered_capacity,
                f"capacity_utilization_{selected_policy}.csv",
                "Download Capacity CSV",
            )


# ============================================================
# ACTION LOG PAGE
# ============================================================

elif page == "Action Log":

    st.subheader("Orchestration Action Log")

    st.write(
        "Review migration and orchestration actions generated by the experiment."
    )

    if action_log.empty:
        st.warning(
            "Action log not found. Run the experiment first."
        )

    else:

        if "policy" in action_log.columns:

            policies = (
                action_log["policy"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_policy = st.selectbox(
                "Filter by policy",
                ["All Policies"] + policies,
            )

            if selected_policy != "All Policies":
                filtered_log = action_log[
                    action_log["policy"].astype(str) == selected_policy
                ]
            else:
                filtered_log = action_log

        else:
            filtered_log = action_log

        st.dataframe(
            filtered_log,
            use_container_width=True,
            hide_index=True,
        )

        show_csv_download(
            filtered_log,
            "orchestration_action_log.csv",
            "Download Action Log CSV",
        )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.subheader("About the Project")

    st.markdown(
        """
        ### Autonomous Multi-Cloud Resource Orchestration

        **Objective:** Compare orchestration policies and evaluate how
        machine-learning-assisted decisions affect service continuity.

        ### Main Components

        1. Workload preprocessing
        2. Random Forest CPU prediction
        3. Isolation Forest anomaly detection
        4. Multi-cloud orchestration simulator
        5. Failure detection and VM migration
        6. SLA and recovery evaluation
        7. Interactive results dashboard

        ### Orchestration Policies

        - **Static:** keeps the initial VM placement.
        - **Reactive:** responds to detected cloud failures and CPU overload.
        - **Risk-Aware:** considers failures, predicted risk, anomaly signals,
          CPU behavior, latency, and projected cloud utilization.

        ### Technologies

        Python, Pandas, Scikit-learn, Streamlit, Matplotlib, and Joblib.

        ### Simulation Context

        The platform evaluates orchestration policies in a simulated
        multi-cloud environment. Cloud capacities, costs, and latencies
        are illustrative; no live cloud accounts are connected.
        """
    )

    st.subheader("Experiment Files")

    files = [
        SUMMARY_FILE,
        OBSERVATIONS_FILE,
        ACTION_LOG_FILE,
        CAPACITY_FILE,
        MODEL_METRICS_FILE,
        VALIDATION_FILE,
        TEST_FILE,
    ]

    for file in files:
        if file.exists():
            st.write(f"✓ {file.relative_to(ROOT)}")
        else:
            st.write(f"— {file.relative_to(ROOT)}")