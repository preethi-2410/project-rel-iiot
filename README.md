# Rel-IIoT: Decentralized Edge-Federated Predictive Maintenance

This repository implements the Rel-IIoT framework, a modular research prototype for decentralized predictive maintenance in Industrial IoT (IIoT) environments. It is designed to meet IEEE conference paper implementation standards, featuring robust evaluation pipelines, security modules, and explainability extensions.

## System Overview
The system features:
1.  **Local Anomaly Detection**: LSTM Autoencoders running on edge nodes with windowed preprocessing.
2.  **Decentralized Federated Learning**: Peer-to-peer parameter exchange without a central server.
3.  **Network Robustness**: Simulation of packet loss, latency, and asynchronous participation.
4.  **Security & Privacy**: Differential Privacy (DP) and Poisoning Defense hooks.
5.  **Explainability**: SHAP-based attribution and runtime cost evaluation.
6.  **Scalability**: Supports varying numbers of nodes, topologies, and communication frequencies.

## Project Structure
- **`src/data`**: Loaders for Synthetic, NASA C-MAPSS, and SECOM datasets with healthy-only split and windowing.
- **`src/models`**: LSTM Autoencoder implementation.
- **`src/federated`**:
    - `node.py`: Edge node logic (training, detection, security, explainability).
    - `system.py`: Simulation engine with `PeerExchange` network layer.
    - `peer_exchange.py`: Network latency and drop rate simulation.
- **`src/security`**:
    - `dp.py`: Differential Privacy (Laplace Mechanism).
    - `poisoning_filter.py`: Statistical outlier defense.
- **`src/evaluation`**: Unified metrics (Precision, Recall, F1, Latency, Divergence) and logging.
- **`experiments`**: Drivers for reproducibility.
- **`config`**: YAML configuration files.

## Setup

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Datasets** (Optional):
    - **NASA C-MAPSS**: Download `FD001.txt` etc. from [NASA PCoE](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). Place in `data/cmapss/`.
    - **SECOM**: Download from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/SECOM). Place `secom.data` in `data/secom/`.
    - *Note*: The system defaults to a robust synthetic generator if files are missing.

## Running Experiments

### Quick Start (All Figures)
```bash
python run_all_experiments.py
```
Generates:
- `results/baselines/local_results.csv`, `results/baselines/centralized_results.csv`
- `results/scalability/scalability_summary.csv` and plots (`scalability.png`, `scalability_heatmap.png`)
- `results/explainability/metrics.csv`, `results/explainability/history.csv`
- `results/explainability_shap/shap_metrics.csv`

### Baselines
```bash
python run_baselines.py
```
Reproduces Figure: Convergence and comparative performance for Local-only vs Centralized.

### Scalability & Ablations
```bash
python run_scalability.py
```
Parameters varied:
- Nodes: 5, 10, 20, 50
- Peers: 2, 5, 10
- Communication frequency (epochs per round): 1, 5, 10
Outputs include communication overhead (`overhead_kb`), round time (`round_time_s`), and model divergence.

### Explainability
```bash
python run_explainability.py
```
Generates stability and runtime metrics; SHAP study is limited to small rounds for practicality.

## Configuration (YAML)
Experiments can be configured via YAML with `experiments/run_experiments.py`:
```bash
python experiments/run_experiments.py --config path/to/config.yaml --output results
```
YAML fields:
- `federated.num_nodes`, `federated.mode`, `federated.connectivity`, `federated.epochs_per_round`, `federated.num_rounds`
- `network.drop_rate`, `network.participation_rate`, `federated.connectivity_prob`
- `security.use_dp`, `security.dp_epsilon`, `security.use_defense`
- `data.dataset_name` (`synthetic`, `cmapss`, `secom`)

## Methodology & Metrics
- **Anomaly Detection**: Reconstruction error (MSE) > Adaptive Threshold. Metrics: Precision, Recall, F1, Detection Latency.
- **Federated Learning**: FedAvg (Centralized) vs. Gossip Averaging (Decentralized). Metrics: Convergence Rate, Model Divergence.
- **Explainability**: SHAP (KernelExplainer) with background summarization. Metrics: Stability, Overlap, Runtime.

## Mapping: Scripts → Paper Sections
- `run_baselines.py` → Baseline comparisons (Local-only vs Centralized FL).
- `run_scalability.py` → Scalability and ablation studies (nodes, peers, frequency).
- `run_explainability.py` → Explainability stability, runtime, SHAP attribution.
- `experiments/run_experiments.py` → Config-driven experiments for reproducibility.

## Hardware/Software Requirements
- Python 3.9+
- Recommended: CPU is sufficient; GPU accelerates LSTM training.
- Packages: `torch`, `numpy`, `pandas`, `networkx`, `seaborn`, `matplotlib`, `shap`, `scikit-learn`.
- Disk space: Minimal; datasets stored under `data/` if used.
