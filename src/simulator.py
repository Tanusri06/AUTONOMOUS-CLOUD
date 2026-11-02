from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "experiment_results"

WORKLOAD_FILE = RESULTS_DIR / "simulation_workloads.csv.gz"
FAILURE_FILE = RESULTS_DIR / "failure_schedule.json"

SUMMARY_FILE = RESULTS_DIR / "orchestration_policy_comparison.csv"
OBSERVATIONS_FILE = RESULTS_DIR / "orchestration_step_observations.csv.gz"
ACTION_LOG_FILE = RESULTS_DIR / "orchestration_action_log.csv"
CAPACITY_FILE = RESULTS_DIR / "cloud_capacity_utilization.csv"

# ============================================================
# SIMULATION CONFIGURATION
# ============================================================

RANDOM_SEED = 42
CPU_OVERLOAD_THRESHOLD = 85.0
SLA_LATENCY_LIMIT_MS = 100.0
MIGRATION_DELAY_STEPS = 1
STEP_MINUTES = 5

# Capacity is aggregate CPU-percentage points across hosted VMs.
# These updated values provide enough simulated headroom for
# Reactive and Risk-Aware policies to migrate VMs.
CLOUDS = {
    "Cloud_A": {"capacity": 320.0, "cost": 1.00, "latency_ms": 25.0},
    "Cloud_B": {"capacity": 360.0, "cost": 0.85, "latency_ms": 45.0},
    "Cloud_C": {"capacity": 280.0, "cost": 0.70, "latency_ms": 65.0},
}

POLICIES = ["Static", "Reactive", "Risk-Aware"]

REQUIRED_COLUMNS = [
    "vm_id",
    "step",
    "actual_cpu",
    "predicted_cpu",
    "memory_usage_kb",
    "disk_read_kbps",
    "disk_write_kbps",
    "network_received_kbps",
    "network_transmitted_kbps",
    "anomaly_flag",
]


# ============================================================
# INPUT LOADING AND VALIDATION
# ============================================================

def load_inputs(workload_path=None, failure_path=None):
    workload_path = Path(workload_path or WORKLOAD_FILE)
    failure_path = Path(failure_path or FAILURE_FILE)

    if not workload_path.exists():
        raise FileNotFoundError(f"Workload file not found:\n{workload_path}")

    if not failure_path.exists():
        raise FileNotFoundError(f"Failure schedule not found:\n{failure_path}")

    workload = pd.read_csv(workload_path)

    with open(failure_path, "r", encoding="utf-8") as file:
        failures = json.load(file)

    missing = [column for column in REQUIRED_COLUMNS if column not in workload.columns]
    if missing:
        raise ValueError(f"Workload file is missing columns: {missing}")

    workload = workload[REQUIRED_COLUMNS].copy()
    workload["vm_id"] = workload["vm_id"].astype(str)
    workload["step"] = pd.to_numeric(workload["step"], errors="raise").astype(int)

    numeric_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in ["vm_id", "step"]
    ]

    for column in numeric_columns:
        workload[column] = pd.to_numeric(workload[column], errors="raise")

    if not np.isfinite(workload[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Workload contains invalid or infinite numeric values.")

    workload = workload.sort_values(["vm_id", "step"]).reset_index(drop=True)

    if workload.empty:
        raise ValueError("The workload file is empty.")

    if workload.duplicated(["vm_id", "step"]).any():
        raise ValueError("Duplicate VM-step combinations were found.")

    step_counts = workload.groupby("vm_id")["step"].nunique()
    if step_counts.nunique() != 1:
        raise ValueError("Every VM must have the same number of steps.")

    expected_steps = int(step_counts.iloc[0])
    expected_sequence = list(range(expected_steps))

    for vm_id, group in workload.groupby("vm_id"):
        if sorted(group["step"].tolist()) != expected_sequence:
            raise ValueError(
                f"VM {vm_id} must have steps 0 to {expected_steps - 1}."
            )

    for cloud, outage in failures.items():
        if cloud not in CLOUDS:
            raise ValueError(f"Unknown cloud in failure schedule: {cloud}")

        if "start_step" not in outage or "end_step" not in outage:
            raise ValueError(f"{cloud} needs start_step and end_step.")

        if (
            outage["start_step"] < 0
            or outage["end_step"] <= outage["start_step"]
        ):
            raise ValueError(f"Invalid failure interval for {cloud}.")

    return workload, failures


# ============================================================
# CLOUD HELPERS
# ============================================================

def cloud_is_healthy(cloud, step, failures):
    outage = failures.get(cloud)

    if outage is None:
        return True

    return not (
        outage["start_step"] <= step < outage["end_step"]
    )


def calculate_cloud_load(placements, vm_cpu):
    loads = {cloud: 0.0 for cloud in CLOUDS}

    for vm_id, cloud in placements.items():
        loads[cloud] += max(0.0, float(vm_cpu.get(vm_id, 0.0)))

    return loads


def calculate_utilization(loads):
    return {
        cloud: 100.0 * loads[cloud] / CLOUDS[cloud]["capacity"]
        for cloud in CLOUDS
    }


def platform_utilization_percent(loads):
    total_load = sum(loads.values())
    total_capacity = sum(info["capacity"] for info in CLOUDS.values())

    if total_capacity <= 0:
        return 0.0

    return 100.0 * total_load / total_capacity


def make_balanced_initial_placements(vm_ids):
    """Create the same balanced starting placement for every policy."""
    cloud_names = list(CLOUDS.keys())

    return {
        vm_id: cloud_names[index % len(cloud_names)]
        for index, vm_id in enumerate(sorted(vm_ids))
    }


def choose_destination(
    vm_id,
    current_cloud,
    vm_cpu,
    placements,
    healthy_clouds,
    policy,
):
    current_cpu = max(0.0, float(vm_cpu.get(vm_id, 0.0)))
    current_loads = calculate_cloud_load(placements, vm_cpu)
    candidates = []

    for cloud in CLOUDS:
        if cloud == current_cloud or not healthy_clouds[cloud]:
            continue

        projected_load = current_loads[cloud] + current_cpu
        capacity = CLOUDS[cloud]["capacity"]

        # Prevent migration into a cloud that would exceed capacity.
        if projected_load > capacity:
            continue

        projected_utilization = 100.0 * projected_load / capacity
        latency = CLOUDS[cloud]["latency_ms"]
        cost = CLOUDS[cloud]["cost"]

        if policy == "Reactive":
            # Reactive prioritizes low latency.
            score = latency
        else:
            # Risk-Aware balances cost, latency, and capacity pressure.
            score = (
                0.45 * cost
                + 0.35 * (latency / 100.0)
                + 0.20 * (projected_utilization / 100.0)
            )

        candidates.append((score, cloud))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def calculate_risk(predicted_cpu, actual_cpu, anomaly_flag):
    cpu_risk = max(float(predicted_cpu), float(actual_cpu)) / 100.0
    anomaly_risk = 1.0 if int(anomaly_flag) == 1 else 0.0

    return min(1.0, 0.70 * cpu_risk + 0.30 * anomaly_risk)


# ============================================================
# RUN ONE POLICY
# ============================================================

def run_policy(policy, workload, failures, initial_placements):
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")

    placements = initial_placements.copy()
    vm_ids = sorted(workload["vm_id"].unique())
    num_steps = int(workload["step"].max()) + 1

    workload_by_step = {
        step: workload[workload["step"] == step].set_index("vm_id")
        for step in range(num_steps)
    }

    observations = []
    actions = []
    capacity_records = []

    migrations = 0
    outage_events = {}

    # vm_id -> {"destination": cloud, "complete_step": step}
    migrating = {}

    for step in range(num_steps):
        step_data = workload_by_step[step]

        actual_cpu = {
            vm_id: float(step_data.loc[vm_id, "actual_cpu"])
            for vm_id in vm_ids
        }

        predicted_cpu = {
            vm_id: float(step_data.loc[vm_id, "predicted_cpu"])
            for vm_id in vm_ids
        }

        anomaly_flags = {
            vm_id: int(step_data.loc[vm_id, "anomaly_flag"])
            for vm_id in vm_ids
        }

        healthy_clouds = {
            cloud: cloud_is_healthy(cloud, step, failures)
            for cloud in CLOUDS
        }

        # Complete migrations scheduled for this step.
        completed_migrations = [
            vm_id
            for vm_id, state in migrating.items()
            if step >= state["complete_step"]
        ]

        for vm_id in completed_migrations:
            placements[vm_id] = migrating[vm_id]["destination"]
            del migrating[vm_id]

        # Record VMs affected when a cloud outage begins.
        for cloud, outage in failures.items():
            if step == outage["start_step"]:
                affected = {
                    vm_id
                    for vm_id in vm_ids
                    if placements[vm_id] == cloud
                }

                outage_events[(cloud, step)] = {
                    "affected": affected,
                    "restored": set(),
                    "start_step": step,
                    "recovery_step": None,
                }

        # ----------------------------------------------------
        # ORCHESTRATION DECISIONS
        # ----------------------------------------------------

        for vm_id in vm_ids:
            if vm_id in migrating:
                continue

            current_cloud = placements[vm_id]
            current_cpu = actual_cpu[vm_id]
            predicted = predicted_cpu[vm_id]
            anomaly = anomaly_flags[vm_id]

            cloud_healthy = healthy_clouds[current_cloud]
            risk = calculate_risk(predicted, current_cpu, anomaly)

            reason = None

            if policy == "Static":
                # Static policy does not initiate migrations.
                continue

            if not cloud_healthy:
                reason = "cloud_failure"

            elif policy == "Reactive":
                if current_cpu >= CPU_OVERLOAD_THRESHOLD:
                    reason = "cpu_overload"

            elif policy == "Risk-Aware":
                if risk >= 0.80:
                    reason = "high_predicted_risk"
                elif current_cpu >= CPU_OVERLOAD_THRESHOLD:
                    reason = "cpu_overload"
                elif anomaly == 1:
                    reason = "anomaly_detected"

            if reason is None:
                continue

            destination = choose_destination(
                vm_id=vm_id,
                current_cloud=current_cloud,
                vm_cpu=actual_cpu,
                placements=placements,
                healthy_clouds=healthy_clouds,
                policy=policy,
            )

            if destination is None:
                actions.append({
                    "policy": policy,
                    "step": step,
                    "vm_id": vm_id,
                    "source_cloud": current_cloud,
                    "destination_cloud": "",
                    "reason": reason,
                    "risk_score": risk,
                    "action": "migration_blocked_no_capacity",
                    "migration_delay_steps": MIGRATION_DELAY_STEPS,
                })
                continue

            migrating[vm_id] = {
                "destination": destination,
                "complete_step": step + MIGRATION_DELAY_STEPS,
            }

            migrations += 1

            actions.append({
                "policy": policy,
                "step": step,
                "vm_id": vm_id,
                "source_cloud": current_cloud,
                "destination_cloud": destination,
                "reason": reason,
                "risk_score": risk,
                "action": "migration_started",
                "migration_delay_steps": MIGRATION_DELAY_STEPS,
            })

        # ----------------------------------------------------
        # RECORD STEP OBSERVATIONS
        # ----------------------------------------------------

        for vm_id in vm_ids:
            assigned_cloud = placements[vm_id]
            assigned_cloud_healthy = healthy_clouds[assigned_cloud]
            is_migrating = vm_id in migrating

            service_available = (
                assigned_cloud_healthy and not is_migrating
            )

            latency_ms = CLOUDS[assigned_cloud]["latency_ms"]

            latency_violation = (
                service_available
                and latency_ms > SLA_LATENCY_LIMIT_MS
            )

            cpu_violation = (
                service_available
                and actual_cpu[vm_id] >= CPU_OVERLOAD_THRESHOLD
            )

            availability_violation = not service_available

            sla_violation = (
                availability_violation
                or latency_violation
                or cpu_violation
            )

            observations.append({
                "policy": policy,
                "step": step,
                "vm_id": vm_id,
                "cloud": assigned_cloud,
                "actual_cpu": actual_cpu[vm_id],
                "predicted_cpu": predicted_cpu[vm_id],
                "risk_score": calculate_risk(
                    predicted_cpu[vm_id],
                    actual_cpu[vm_id],
                    anomaly_flags[vm_id],
                ),
                "anomaly_flag": anomaly_flags[vm_id],
                "cloud_healthy": assigned_cloud_healthy,
                "is_migrating": is_migrating,
                "service_available": service_available,
                "latency_ms": latency_ms,
                "availability_violation": availability_violation,
                "latency_violation": latency_violation,
                "cpu_violation": cpu_violation,
                "sla_violation": sla_violation,
                "migration": is_migrating,
            })

            # Recovery is complete when every affected VM is available.
            for event in outage_events.values():
                if vm_id not in event["affected"]:
                    continue

                if service_available:
                    event["restored"].add(vm_id)

                if (
                    event["affected"]
                    and event["affected"] <= event["restored"]
                    and event["recovery_step"] is None
                ):
                    event["recovery_step"] = step

        # ----------------------------------------------------
        # CAPACITY RECORDS
        # ----------------------------------------------------

        loads = calculate_cloud_load(placements, actual_cpu)
        utilization = calculate_utilization(loads)
        platform_util = platform_utilization_percent(loads)

        for cloud in CLOUDS:
            capacity_records.append({
                "policy": policy,
                "step": step,
                "cloud": cloud,
                "capacity": CLOUDS[cloud]["capacity"],
                "cpu_load": loads[cloud],
                "utilization_percent": utilization[cloud],
                "platform_utilization_percent": platform_util,
                "cloud_healthy": healthy_clouds[cloud],
            })

    # ========================================================
    # METRICS
    # ========================================================

    observations_df = pd.DataFrame(observations)

    actions_df = pd.DataFrame(actions, columns=[
        "policy",
        "step",
        "vm_id",
        "source_cloud",
        "destination_cloud",
        "reason",
        "risk_score",
        "action",
        "migration_delay_steps",
    ])

    capacity_df = pd.DataFrame(capacity_records)

    total_vm_steps = len(observations_df)
    available_vm_steps = int(observations_df["service_available"].sum())
    unavailable_vm_steps = int(
        observations_df["availability_violation"].sum()
    )

    sla_violations = int(observations_df["sla_violation"].sum())
    latency_violations = int(observations_df["latency_violation"].sum())
    cpu_violations = int(observations_df["cpu_violation"].sum())

    availability_percent = (
        100.0 * available_vm_steps / total_vm_steps
    )

    sla_violation_rate_percent = (
        100.0 * sla_violations / total_vm_steps
    )

    unavailable_rate_percent = (
        100.0 * unavailable_vm_steps / total_vm_steps
    )

    performance_sla_violations = (
        latency_violations + cpu_violations
    )

    performance_sla_violation_rate_percent = (
        100.0 * performance_sla_violations / total_vm_steps
    )

    # Platform utilization = total CPU load / total cloud capacity.
    mean_platform_utilization = (
        capacity_df.groupby("step")[
            "platform_utilization_percent"
        ].first().mean()
    )

    # Average individual cloud utilization, kept as a diagnostic.
    mean_cloud_utilization = (
        capacity_df["utilization_percent"].mean()
    )

    # Illustrative cost only; this is not actual provider billing.
    illustrative_cost = sum(
        CLOUDS[row["cloud"]]["cost"]
        for _, row in observations_df.iterrows()
        if row["service_available"]
    )

    recovery_durations = []

    for event in outage_events.values():
        if not event["affected"]:
            continue

        if event["recovery_step"] is not None:
            duration_steps = (
                event["recovery_step"] - event["start_step"]
            )
        else:
            duration_steps = (
                num_steps - event["start_step"]
            )

        recovery_durations.append(max(0, duration_steps))

    mean_recovery_steps = (
        float(np.mean(recovery_durations))
        if recovery_durations
        else 0.0
    )

    metrics = {
        "policy": policy,
        "availability_percent": availability_percent,
        "sla_violation_rate_percent": sla_violation_rate_percent,
        "unavailable_vm_steps": unavailable_vm_steps,
        "unavailable_rate_percent": unavailable_rate_percent,
        "performance_sla_violations": performance_sla_violations,
        "performance_sla_violation_rate_percent":
            performance_sla_violation_rate_percent,
        "latency_violations": latency_violations,
        "cpu_violations": cpu_violations,
        "migrations": migrations,
        "mean_recovery_time_steps": mean_recovery_steps,
        "mean_recovery_time_minutes": (
            mean_recovery_steps * STEP_MINUTES
        ),
        "mean_capacity_utilization_percent": (
            mean_platform_utilization
        ),
        "mean_cloud_utilization_percent": (
            mean_cloud_utilization
        ),
        "illustrative_cost_units": illustrative_cost,
        "total_vm_steps": total_vm_steps,
        "available_vm_steps": available_vm_steps,
        "sla_violations": sla_violations,
    }

    return metrics, observations_df, actions_df, capacity_df


# ============================================================
# RUN ALL POLICIES
# ============================================================

def run_experiment(workload, failures, output_dir=None):
    """
    Run all three policies using the same workload and initial placements.

    output_dir can be a unique folder for each dashboard experiment.
    """
    output_dir = Path(output_dir or RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    workload = workload.copy()
    workload["vm_id"] = workload["vm_id"].astype(str)
    workload["step"] = workload["step"].astype(int)

    workload = workload.sort_values(
        ["vm_id", "step"]
    ).reset_index(drop=True)

    vm_ids = sorted(workload["vm_id"].unique())

    if not vm_ids:
        raise ValueError(
            "Cannot run experiment: workload contains no VMs."
        )

    num_steps = int(workload["step"].max()) + 1

    # Identical balanced initial placement for every policy.
    initial_placements = make_balanced_initial_placements(vm_ids)

    print(f"VMs: {len(vm_ids)}")
    print(f"Steps per VM: {num_steps}")
    print("All policies use the same balanced initial placement.")

    all_metrics = []
    all_observations = []
    all_actions = []
    all_capacity = []

    for policy in POLICIES:
        print(f"\nRunning policy: {policy}")

        metrics, observations, actions, capacity = run_policy(
            policy,
            workload,
            failures,
            initial_placements,
        )

        all_metrics.append(metrics)
        all_observations.append(observations)
        all_actions.append(actions)
        all_capacity.append(capacity)

        print(
            f"Availability: "
            f"{metrics['availability_percent']:.2f}%"
        )
        print(
            f"SLA violation rate: "
            f"{metrics['sla_violation_rate_percent']:.2f}%"
        )
        print(f"Migrations: {metrics['migrations']}")
        print(
            f"Mean recovery: "
            f"{metrics['mean_recovery_time_minutes']:.2f} min"
        )

    summary_df = pd.DataFrame(all_metrics)

    observations_df = pd.concat(
        all_observations,
        ignore_index=True,
    )

    actions_df = pd.concat(
        all_actions,
        ignore_index=True,
    )

    capacity_df = pd.concat(
        all_capacity,
        ignore_index=True,
    )

    summary_df.to_csv(
        output_dir / "orchestration_policy_comparison.csv",
        index=False,
    )

    observations_df.to_csv(
        output_dir / "orchestration_step_observations.csv.gz",
        index=False,
        compression="gzip",
    )

    actions_df.to_csv(
        output_dir / "orchestration_action_log.csv",
        index=False,
    )

    capacity_df.to_csv(
        output_dir / "cloud_capacity_utilization.csv",
        index=False,
    )

    display_columns = [
        "policy",
        "availability_percent",
        "sla_violation_rate_percent",
        "unavailable_vm_steps",
        "performance_sla_violations",
        "migrations",
        "mean_recovery_time_minutes",
        "mean_capacity_utilization_percent",
        "mean_cloud_utilization_percent",
        "illustrative_cost_units",
    ]

    print("\nSIMULATION COMPLETED")
    print(summary_df[display_columns].to_string(index=False))
    print(f"\nResults saved to: {output_dir}")

    return (
        summary_df,
        observations_df,
        actions_df,
        capacity_df,
    )


# ============================================================
# ORIGINAL COMMAND-LINE WORKFLOW
# ============================================================

def main():
    print("=" * 65)
    print("AUTONOMOUS MULTI-CLOUD ORCHESTRATION SIMULATOR")
    print("=" * 65)

    workload, failures = load_inputs()

    print(f"\nWorkload rows: {len(workload)}")
    print(f"VMs: {workload['vm_id'].nunique()}")
    print(
        f"Steps per VM: "
        f"{workload.groupby('vm_id')['step'].nunique().iloc[0]}"
    )

    run_experiment(
        workload,
        failures,
        output_dir=RESULTS_DIR,
    )


if __name__ == "__main__":
    main()