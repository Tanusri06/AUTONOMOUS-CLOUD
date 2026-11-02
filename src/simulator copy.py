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

NUM_VMS = 30
NUM_STEPS = 288
RANDOM_SEED = 42

CPU_OVERLOAD_THRESHOLD = 85.0
SLA_LATENCY_LIMIT_MS = 100.0
MIGRATION_DELAY_STEPS = 1

CLOUDS = {
    "Cloud_A": {
        "capacity": 160.0,
        "cost": 1.00,
        "latency_ms": 25.0,
    },
    "Cloud_B": {
        "capacity": 180.0,
        "cost": 0.85,
        "latency_ms": 45.0,
    },
    "Cloud_C": {
        "capacity": 140.0,
        "cost": 0.70,
        "latency_ms": 65.0,
    },
}

POLICIES = ["Static", "Reactive", "Risk-Aware"]


# ============================================================
# LOAD AND VALIDATE INPUTS
# ============================================================

def load_inputs():
    if not WORKLOAD_FILE.exists():
        raise FileNotFoundError(
            f"Workload file not found:\n{WORKLOAD_FILE}\n"
            "Run: python src/prepare_simulation.py"
        )

    if not FAILURE_FILE.exists():
        raise FileNotFoundError(
            f"Failure schedule not found:\n{FAILURE_FILE}\n"
            "Run: python src/prepare_simulation.py"
        )

    workload = pd.read_csv(WORKLOAD_FILE)

    with open(FAILURE_FILE, "r", encoding="utf-8") as file:
        failures = json.load(file)

    required_columns = [
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

    missing = [
        column for column in required_columns
        if column not in workload.columns
    ]

    if missing:
        raise ValueError(
            f"Workload file is missing columns: {missing}"
        )

    workload["vm_id"] = workload["vm_id"].astype(str)
    workload["step"] = workload["step"].astype(int)

    workload = workload.sort_values(
        ["vm_id", "step"]
    ).reset_index(drop=True)

    vm_count = workload["vm_id"].nunique()

    if vm_count != NUM_VMS:
        raise ValueError(
            f"Expected {NUM_VMS} VMs, found {vm_count}."
        )

    step_counts = workload.groupby("vm_id")["step"].nunique()

    if not (step_counts == NUM_STEPS).all():
        raise ValueError(
            f"Each VM must have exactly {NUM_STEPS} steps."
        )

    return workload, failures


# ============================================================
# CLOUD AND RESOURCE HELPERS
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
        loads[cloud] += max(
            0.0,
            float(vm_cpu.get(vm_id, 0.0))
        )

    return loads


def calculate_utilization(loads):
    return {
        cloud: 100.0 * loads[cloud] / CLOUDS[cloud]["capacity"]
        for cloud in CLOUDS
    }


def choose_destination(
    vm_id,
    current_cloud,
    vm_cpu,
    placements,
    healthy_clouds,
    policy,
):
    current_cpu = max(
        0.0,
        float(vm_cpu.get(vm_id, 0.0))
    )

    current_loads = calculate_cloud_load(
        placements,
        vm_cpu,
    )

    candidates = []

    for cloud in CLOUDS:

        if cloud == current_cloud:
            continue

        if not healthy_clouds[cloud]:
            continue

        projected_load = current_loads[cloud] + current_cpu
        capacity = CLOUDS[cloud]["capacity"]

        if projected_load > capacity:
            continue

        projected_utilization = (
            100.0 * projected_load / capacity
        )

        latency = CLOUDS[cloud]["latency_ms"]
        cost = CLOUDS[cloud]["cost"]

        if policy == "Reactive":
            score = latency

        else:
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
    cpu_risk = max(
        float(predicted_cpu),
        float(actual_cpu),
    ) / 100.0

    anomaly_risk = (
        1.0 if int(anomaly_flag) == 1 else 0.0
    )

    return min(
        1.0,
        0.70 * cpu_risk + 0.30 * anomaly_risk
    )


# ============================================================
# RUN ONE ORCHESTRATION POLICY
# ============================================================

def run_policy(policy, workload, failures, initial_placements):

    placements = initial_placements.copy()

    vm_ids = sorted(workload["vm_id"].unique())

    workload_by_step = {
        step: workload[
            workload["step"] == step
        ].set_index("vm_id")
        for step in range(NUM_STEPS)
    }

    observations = []
    actions = []
    capacity_records = []

    migrations = 0

    # Recovery tracking:
    # For each cloud outage, record how many steps pass until
    # every VM originally placed on that cloud is restored.
    recovery_events = []

    # Track original VMs affected by each cloud failure.
    outage_affected_vms = {}

    for step in range(NUM_STEPS):

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

        # ----------------------------------------------------
        # Record VMs affected when a cloud outage begins
        # ----------------------------------------------------

        for cloud, outage in failures.items():

            if step == outage["start_step"]:

                affected = [
                    vm_id
                    for vm_id in vm_ids
                    if placements[vm_id] == cloud
                ]

                outage_affected_vms[(cloud, step)] = {
                    "affected_vms": set(affected),
                    "restored_vms": set(),
                    "start_step": step,
                }

        # ----------------------------------------------------
        # Process every VM
        # ----------------------------------------------------

        for vm_id in vm_ids:

            current_cloud = placements[vm_id]

            current_cpu = actual_cpu[vm_id]
            predicted = predicted_cpu[vm_id]
            anomaly = anomaly_flags[vm_id]

            current_cloud_healthy = healthy_clouds[current_cloud]

            risk = calculate_risk(
                predicted,
                current_cpu,
                anomaly,
            )

            migration_reason = None

            # ---------------- Static ----------------

            if policy == "Static":
                migration_reason = None

            # ---------------- Reactive ----------------

            elif policy == "Reactive":

                if not current_cloud_healthy:
                    migration_reason = "cloud_failure"

                elif current_cpu >= CPU_OVERLOAD_THRESHOLD:
                    migration_reason = "cpu_overload"

            # ---------------- Risk-Aware ----------------

            elif policy == "Risk-Aware":

                if not current_cloud_healthy:
                    migration_reason = "cloud_failure"

                elif risk >= 0.80:
                    migration_reason = "high_risk"

                elif (
                    predicted >= CPU_OVERLOAD_THRESHOLD
                    or current_cpu >= CPU_OVERLOAD_THRESHOLD
                    or anomaly == 1
                ):
                    migration_reason = "risk_signal"

            # ------------------------------------------------
            # Migration decision and action
            # ------------------------------------------------

            migrated = False
            migration_reason_record = None

            if migration_reason is not None:

                destination = choose_destination(
                    vm_id=vm_id,
                    current_cloud=current_cloud,
                    vm_cpu=actual_cpu,
                    placements=placements,
                    healthy_clouds=healthy_clouds,
                    policy=policy,
                )

                if destination is not None:

                    actions.append({
                        "policy": policy,
                        "step": step,
                        "vm_id": vm_id,
                        "source_cloud": current_cloud,
                        "destination_cloud": destination,
                        "reason": migration_reason,
                        "risk_score": risk,
                        "migration_delay_steps": MIGRATION_DELAY_STEPS,
                    })

                    placements[vm_id] = destination
                    migrations += 1

                    migrated = True
                    migration_reason_record = migration_reason

            # ------------------------------------------------
            # Service and SLA evaluation
            # ------------------------------------------------

            assigned_cloud = placements[vm_id]
            assigned_cloud_healthy = healthy_clouds[assigned_cloud]

            # A VM is unavailable if:
            # 1. Its assigned cloud is down, or
            # 2. It is undergoing the simulated migration delay.
            service_available = (
                assigned_cloud_healthy
                and not migrated
            )

            if assigned_cloud_healthy:
                latency_ms = CLOUDS[assigned_cloud]["latency_ms"]
            else:
                latency_ms = np.nan

            latency_violation = (
                service_available
                and latency_ms > SLA_LATENCY_LIMIT_MS
            )

            cpu_violation = (
                service_available
                and current_cpu >= CPU_OVERLOAD_THRESHOLD
            )

            # IMPORTANT SLA FIX:
            # Any unavailable VM-step is an SLA violation.
            # Performance violations are counted separately.
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
                "actual_cpu": current_cpu,
                "predicted_cpu": predicted,
                "risk_score": risk,
                "anomaly_flag": anomaly,
                "cloud_healthy": assigned_cloud_healthy,
                "service_available": service_available,
                "latency_ms": latency_ms,
                "availability_violation": availability_violation,
                "latency_violation": latency_violation,
                "cpu_violation": cpu_violation,
                "sla_violation": sla_violation,
                "migration": migrated,
                "migration_reason": migration_reason_record,
            })

            # ------------------------------------------------
            # Track recovery after outage
            # ------------------------------------------------

            for key, event in outage_affected_vms.items():

                cloud, outage_start = key

                if vm_id not in event["affected_vms"]:
                    continue

                if service_available:
                    event["restored_vms"].add(vm_id)

                if (
                    event["affected_vms"]
                    and event["affected_vms"]
                    <= event["restored_vms"]
                    and "recovery_step" not in event
                ):
                    event["recovery_step"] = step

        # ----------------------------------------------------
        # Capacity utilization record
        # ----------------------------------------------------

        loads = calculate_cloud_load(
            placements,
            actual_cpu,
        )

        utilization = calculate_utilization(loads)

        for cloud in CLOUDS:

            capacity_records.append({
                "policy": policy,
                "step": step,
                "cloud": cloud,
                "capacity": CLOUDS[cloud]["capacity"],
                "cpu_load": loads[cloud],
                "utilization_percent": utilization[cloud],
                "cloud_healthy": healthy_clouds[cloud],
            })

    # ========================================================
    # METRICS
    # ========================================================

    observations_df = pd.DataFrame(observations)
    actions_df = pd.DataFrame(actions)
    capacity_df = pd.DataFrame(capacity_records)

    total_vm_steps = len(observations_df)

    available_vm_steps = int(
        observations_df["service_available"].sum()
    )

    unavailable_vm_steps = int(
        observations_df["availability_violation"].sum()
    )

    sla_violations = int(
        observations_df["sla_violation"].sum()
    )

    latency_violations = int(
        observations_df["latency_violation"].sum()
    )

    cpu_violations = int(
        observations_df["cpu_violation"].sum()
    )

    availability_percent = (
        available_vm_steps / total_vm_steps * 100.0
    )

    sla_violation_rate_percent = (
        sla_violations / total_vm_steps * 100.0
    )

    unavailable_rate_percent = (
        unavailable_vm_steps / total_vm_steps * 100.0
    )

    performance_sla_violations = (
        latency_violations + cpu_violations
    )

    performance_sla_violation_rate_percent = (
        performance_sla_violations / total_vm_steps * 100.0
    )

    mean_capacity_utilization = (
        capacity_df["utilization_percent"].mean()
    )

    # Illustrative cost only; not actual provider billing.
    illustrative_cost = 0.0

    for _, row in observations_df.iterrows():

        if row["service_available"]:
            illustrative_cost += CLOUDS[row["cloud"]]["cost"]

    # Calculate actual recovery duration from simulated outage start
    # to restoration of all VMs originally affected by that outage.
    recovery_durations = []

    for event in outage_affected_vms.values():

        if "recovery_step" in event:
            duration = (
                event["recovery_step"]
                - event["start_step"]
            )
        else:
            duration = NUM_STEPS - event["start_step"]

        recovery_durations.append(duration)

    mean_recovery_time_steps = (
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
        "mean_recovery_time_steps": mean_recovery_time_steps,
        "mean_recovery_time_minutes":
            mean_recovery_time_steps * 5.0,
        "mean_capacity_utilization_percent":
            mean_capacity_utilization,
        "illustrative_cost_units": illustrative_cost,
        "total_vm_steps": total_vm_steps,
        "available_vm_steps": available_vm_steps,
        "sla_violations": sla_violations,
    }

    return (
        metrics,
        observations_df,
        actions_df,
        capacity_df,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 65)
    print("AUTONOMOUS MULTI-CLOUD ORCHESTRATION SIMULATOR")
    print("=" * 65)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    workload, failures = load_inputs()

    print(f"\nWorkload rows: {len(workload)}")
    print(f"VMs: {workload['vm_id'].nunique()}")
    print(f"Steps per VM: {NUM_STEPS}")

    # Same initial placement for all policies
    rng = np.random.default_rng(RANDOM_SEED)

    vm_ids = sorted(workload["vm_id"].unique())
    cloud_names = list(CLOUDS.keys())

    initial_placements = {
        vm_id: cloud_names[
            rng.integers(0, len(cloud_names))
        ]
        for vm_id in vm_ids
    }

    print("\nAll policies use the same initial VM placement.")

    all_metrics = []
    all_observations = []
    all_actions = []
    all_capacity = []

    for policy in POLICIES:

        print("\n" + "-" * 65)
        print(f"Running policy: {policy}")
        print("-" * 65)

        (
            metrics,
            observations,
            actions,
            capacity,
        ) = run_policy(
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

        print(
            f"Unavailable VM-steps: "
            f"{metrics['unavailable_vm_steps']}"
        )

        print(
            f"Performance SLA violations: "
            f"{metrics['performance_sla_violations']}"
        )

        print(f"Migrations: {metrics['migrations']}")

        print(
            f"Mean recovery time: "
            f"{metrics['mean_recovery_time_minutes']:.2f} minutes"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

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
        SUMMARY_FILE,
        index=False,
    )

    observations_df.to_csv(
        OBSERVATIONS_FILE,
        index=False,
        compression="gzip",
    )

    actions_df.to_csv(
        ACTION_LOG_FILE,
        index=False,
    )

    capacity_df.to_csv(
        CAPACITY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SIMULATION COMPLETED")
    print("=" * 65)

    display_columns = [
        "policy",
        "availability_percent",
        "sla_violation_rate_percent",
        "unavailable_vm_steps",
        "performance_sla_violations",
        "migrations",
        "mean_recovery_time_minutes",
        "mean_capacity_utilization_percent",
        "illustrative_cost_units",
    ]

    print(
        summary_df[display_columns].to_string(index=False)
    )

    print("\nSaved output files:")
    print(SUMMARY_FILE)
    print(OBSERVATIONS_FILE)
    print(ACTION_LOG_FILE)
    print(CAPACITY_FILE)


if __name__ == "__main__":
    main()