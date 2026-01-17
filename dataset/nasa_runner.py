import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.federated.system import FederatedSimulation

def run_nasa_local(output_dir='results/datasets/nasa', rounds=10, epochs=1, nodes=5):
    os.makedirs(output_dir, exist_ok=True)
    sim = FederatedSimulation(
        num_nodes=nodes,
        dataset_name='cmapss',
        mode='local',
        connectivity='ring'
    )
    history = sim.run_rounds(rounds=rounds, epochs_per_round=epochs)
    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'local_metrics.csv'), index=False)

if __name__ == '__main__':
    run_nasa_local()
