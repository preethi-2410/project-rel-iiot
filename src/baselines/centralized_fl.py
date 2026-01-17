import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.federated.system import FederatedSimulation

def run_centralized_baseline(output_dir='results/baselines', dataset='synthetic'):
    print("Running Centralized FL Baseline...")
    sim = FederatedSimulation(
        num_nodes=5, 
        dataset_name=dataset, 
        mode='centralized'
    )
    history = sim.run_rounds(rounds=10, epochs_per_round=1)
    
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'centralized_results.csv'), index=False)
    print("Centralized Baseline Done.")

if __name__ == "__main__":
    run_centralized_baseline()
