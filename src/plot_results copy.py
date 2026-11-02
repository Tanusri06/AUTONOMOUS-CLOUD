from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "experiment_results"
GRAPH_DIR = RESULTS_DIR / "graphs"

SUMMARY_FILE = RESULTS_DIR / "orchestration_policy_comparison.csv"
CAPACITY_FILE = RESULTS_DIR / "cloud_capacity_utilization.csv"


# ============================================================
# GRAPH STYLE
# ============================================================

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
})


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
            "Run the simulator first: python src/simulator.py"
        )

    return pd.read_csv(path)


def save_bar_chart(
    data,
    x_column,
    y_column,
    title,
    y_label,
    filename,
    percentage=False,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        data[x_column].astype(str),
        data[y_column],
        width=0.62,
    )

    ax.set_title(title, pad=20)
    ax.set_xlabel("Orchestration Policy", labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    ax.set_axisbelow(True)

    # Add space above the tallest bar so labels don't hit the title.
    max_value = float(data[y_column].max())

    if percentage:
        ax.set_ylim(0, 105)
    elif max_value > 0:
        ax.set_ylim(0, max_value * 1.18)
    else:
        ax.set_ylim(0, 1)

    # Add readable value labels above each bar.
    for bar in bars:
        value = bar.get_height()

        label = (
            f"{value:.2f}%"
            if percentage
            else f"{value:,.2f}"
        )

        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Created: {filename}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GENERATING ORCHESTRATION RESULT GRAPHS")
    print("=" * 60)

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = load_csv(SUMMARY_FILE)
    capacity = load_csv(CAPACITY_FILE)

    required_columns = [
        "policy",
        "availability_percent",
        "sla_violation_rate_percent",
        "migrations",
        "mean_recovery_time_minutes",
        "mean_capacity_utilization_percent",
        "illustrative_cost_units",
    ]

    missing = [
        column for column in required_columns
        if column not in summary.columns
    ]

    if missing:
        raise ValueError(
            f"Summary CSV is missing columns: {missing}"
        )

    policy_order = [
        "Static",
        "Reactive",
        "Risk-Aware",
    ]

    summary["policy"] = pd.Categorical(
        summary["policy"],
        categories=policy_order,
        ordered=True,
    )

    summary = summary.sort_values("policy")

    # --------------------------------------------------------
    # 1. Availability
    # --------------------------------------------------------

    save_bar_chart(
        summary,
        "policy",
        "availability_percent",
        "Availability Comparison Across Policies",
        "Availability (%)",
        "01_availability_comparison.png",
        percentage=True,
    )

    # --------------------------------------------------------
    # 2. SLA violation rate
    # --------------------------------------------------------

    save_bar_chart(
        summary,
        "policy",
        "sla_violation_rate_percent",
        "SLA Violation Rate Across Policies",
        "SLA Violation Rate (%)",
        "02_sla_violation_comparison.png",
    )

    # --------------------------------------------------------
    # 3. Migrations
    # --------------------------------------------------------

    save_bar_chart(
        summary,
        "policy",
        "migrations",
        "Total VM Migrations Across Policies",
        "Number of Migrations",
        "03_migration_comparison.png",
    )

    # --------------------------------------------------------
    # 4. Recovery time
    # --------------------------------------------------------

    save_bar_chart(
        summary,
        "policy",
        "mean_recovery_time_minutes",
        "Mean Simulated Recovery Time",
        "Recovery Time (minutes)",
        "04_recovery_time_comparison.png",
    )

    # --------------------------------------------------------
    # 5. Capacity utilization
    # --------------------------------------------------------

    save_bar_chart(
        summary,
        "policy",
        "mean_capacity_utilization_percent",
        "Mean Cloud Capacity Utilization",
        "Capacity Utilization (%)",
        "05_capacity_utilization_comparison.png",
    )

    # --------------------------------------------------------
    # 6. Illustrative cost
    # --------------------------------------------------------

    save_bar_chart(
        summary,
        "policy",
        "illustrative_cost_units",
        "Illustrative Cost Comparison",
        "Illustrative Cost Units",
        "06_illustrative_cost_comparison.png",
    )

    # --------------------------------------------------------
    # 7. Capacity utilization over time
    # --------------------------------------------------------

    required_capacity_columns = [
        "policy",
        "step",
        "cloud",
        "utilization_percent",
    ]

    missing = [
        column for column in required_capacity_columns
        if column not in capacity.columns
    ]

    if missing:
        raise ValueError(
            f"Capacity CSV is missing columns: {missing}"
        )

    for policy in policy_order:

        policy_data = capacity[
            capacity["policy"] == policy
        ]

        if policy_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        for cloud in sorted(policy_data["cloud"].unique()):

            cloud_data = policy_data[
                policy_data["cloud"] == cloud
            ].sort_values("step")

            ax.plot(
                cloud_data["step"],
                cloud_data["utilization_percent"],
                label=cloud,
                linewidth=1.6,
            )

        ax.set_title(
            f"Cloud Capacity Utilization Over Time — {policy}",
            pad=18,
        )

        ax.set_xlabel("Simulation Step (5-minute intervals)")
        ax.set_ylabel("Capacity Utilization (%)")

        ax.legend(title="Cloud")
        ax.grid(True, linestyle="--", alpha=0.35)

        fig.tight_layout()

        filename = (
            f"07_capacity_over_time_"
            f"{policy.lower().replace('-', '_')}.png"
        )

        fig.savefig(
            GRAPH_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Created: {filename}")

    # --------------------------------------------------------
    # 8. SLA components
    # --------------------------------------------------------

    sla_columns = [
        "unavailable_vm_steps",
        "latency_violations",
        "cpu_violations",
    ]

    if all(column in summary.columns for column in sla_columns):

        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(len(summary))
        width = 0.24

        ax.bar(
            [i - width for i in x],
            summary["unavailable_vm_steps"],
            width,
            label="Unavailable VM-steps",
        )

        ax.bar(
            x,
            summary["latency_violations"],
            width,
            label="Latency violations",
        )

        ax.bar(
            [i + width for i in x],
            summary["cpu_violations"],
            width,
            label="CPU violations",
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels(summary["policy"].astype(str))

        ax.set_title("SLA Violation Components by Policy", pad=18)
        ax.set_xlabel("Orchestration Policy")
        ax.set_ylabel("Violation Count")

        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        fig.tight_layout()

        fig.savefig(
            GRAPH_DIR / "08_sla_components_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print("Created: 08_sla_components_comparison.png")

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GRAPH GENERATION COMPLETED")
    print("=" * 60)

    print(f"\nGraphs saved in:\n{GRAPH_DIR}")

    print("\nGenerated files:")

    for file in sorted(GRAPH_DIR.glob("*.png")):
        print(f"- {file.name}")


if __name__ == "__main__":
    main()