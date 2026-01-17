# Rel-IIoT: Decentralized Edge-Federated Predictive Maintenance

This repository implements the Rel-IIoT framework, a modular research prototype for decentralized predictive maintenance in Industrial IoT (IIoT) environments.

## System Overview
The system features:
1.  **Local Anomaly Detection**: LSTM Autoencoders running on edge nodes.
2.  **Decentralized Federated Learning**: Peer-to-peer parameter exchange without a central server.
3.  **On-Device Explainability**: Gradient-based attribution for anomalies.
4.  **Scalability**: Supports varying numbers of nodes and topologies.

## Architecture
- **`src/data`**: Data loaders for Synthetic, C-MAPSS, and SECOM datasets.
- **`src/models`**: LSTM Autoencoder implementation in PyTorch.
- **`src/federated`**:
    - `node.py`: Edge node logic (training, detection, explainability).
    - `system.py`: Simulation engine for federated rounds and topology management.
- **`src/explainability`**: Engine for interpreting model decisions (Saliency maps, SHAP).
- **`experiments`**: Scripts for running validation studies.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  (Optional) If you have C-MAPSS or SECOM datasets, place them in `data/`. The system defaults to a robust synthetic generator if files are missing.

## Running Experiments

To reproduce the results (Convergence, Baselines, Scalability):

```bash
python experiments/run_experiments.py --output results
```

This will generate:
- `results/convergence.png`: F1 score over rounds for Decentralized vs Centralized vs Local.
- `results/scalability.png`: Performance stability across node counts (5, 10, 20).
- `results/baseline_results.csv`: Raw metrics.

## Methodology

### Model
- **Type**: LSTM Autoencoder
- **Input**: Sliding window of multivariate sensor data.
- **Anomaly Detection**: Reconstruction error (MSE) > Adaptive Threshold (Mean + 3*StdDev).

### Federated Learning
- **Protocol**: Decentralized Peer-to-Peer (P2P).
- **Aggregation**: Weighted parameter averaging with neighbors (Ring topology default).
- **Baselines**:
    - *Local-only*: Nodes train in isolation.
    - *Centralized*: Simulated central server aggregation.

### Explainability
- **Technique**: Gradient x Input (Saliency) to identify sensors contributing most to the reconstruction error.

## Results
Experiment outputs are saved in the `results/` directory. The prototype demonstrates that the decentralized approach achieves comparable performance to centralized methods while preserving data privacy and eliminating single points of failure.
