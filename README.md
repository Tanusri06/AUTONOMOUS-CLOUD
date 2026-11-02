AUTONOMOUS MULTI-CLOUD RESOURCE ORCHESTRATION AND SELF-HEALING INFRASTRUCTURE PLATFORM

OVERVIEW

This project presents an ML-assisted multi-cloud resource orchestration and self-healing platform designed to evaluate workload management strategies in simulated cloud environments.

The platform uses real cloud workload traces from the Bitbrains dataset to analyze resource usage, predict CPU workload, detect anomalies, and evaluate orchestration decisions under changing workload conditions and simulated failures.

It compares three orchestration policies—Static, Reactive, and Risk-Aware—across multiple simulated cloud environments with different capacities, costs, and latency characteristics.

The system is implemented as an interactive Streamlit dashboard that allows users to upload workload data, run experiments, visualize performance metrics, and download experiment results.

OBJECTIVES

- Analyze cloud workload behavior using real workload traces.
- Apply machine learning for CPU workload prediction and anomaly detection.
- Simulate multi-cloud resource orchestration and self-healing decisions.
- Compare Static, Reactive, and Risk-Aware orchestration policies.
- Evaluate system performance under simulated failures and workload changes.
- Visualize results through an interactive dashboard.

KEY FEATURES

- Workload Prediction: Uses a Random Forest model to predict CPU workload.
- Anomaly Detection: Uses an Isolation Forest model to identify unusual workload behavior.
- Multi-Cloud Simulation: Models multiple cloud environments with different resource capacities, costs, and latency.
- Orchestration Policy Comparison: Evaluates Static, Reactive, and Risk-Aware policies.
- Self-Healing Simulation: Models failure detection, recovery, and migration decisions.
- Performance Evaluation: Measures availability, SLA violations, migration count, recovery time, and resource utilization.
- Interactive Dashboard: Supports workload CSV uploads, experiment execution, graphs, performance summaries, and CSV downloads.

TECHNOLOGIES USED

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Joblib

SYSTEM WORKFLOW

1. Upload a prepared cloud workload dataset.
2. Process workload information and extract relevant features.
3. Apply trained machine-learning models for prediction and anomaly detection.
4. Simulate workload conditions and cloud failures.
5. Execute the selected orchestration policy.
6. Record migration, recovery, and resource-management decisions.
7. Compare performance metrics and visualize the results in the dashboard.

ORCHESTRATION POLICIES

Static Policy  
Maintains the initial workload placement without proactive orchestration responses.

Reactive Policy  
Responds to detected workload pressure or failures by attempting migration or recovery actions.

Risk-Aware Policy  
Considers workload and cloud conditions before selecting orchestration actions.

PERFORMANCE METRICS

The platform evaluates:

- Service availability
- SLA violations
- Migration count
- Recovery time
- Cloud capacity utilization

PROJECT SCOPE AND LIMITATIONS

The current implementation focuses on autonomous orchestration decision-making and policy evaluation within a simulated multi-cloud environment.

The workload traces are based on real cloud measurements, while cloud capacities, infrastructure conditions, migration actions, and failure scenarios are simulated. The platform does not currently provision or migrate live virtual machines across cloud providers.

Actual cloud deployment, live infrastructure telemetry, and execution of migration or recovery actions on cloud resources are potential future extensions.

RUNNING THE DASHBOARD LOCALLY

Install the required dependencies:

    pip install -r requirements.txt

Run the Streamlit dashboard:

    streamlit run dashboard/app.py

PROJECT PURPOSE

The project demonstrates how machine learning and orchestration policies can support automated resource management and self-healing decisions in multi-cloud environments. It evaluates the impact of different strategies on availability, SLA compliance, recovery, and resource utilization.
