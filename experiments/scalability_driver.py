import itertools
import os
import pandas as pd
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.federated.system import FederatedSimulation

def run_scalability_study(output_dir='results/scalability'):
    os.makedirs(output_dir, exist_ok=True)
    
    nodes_list = [5, 10, 20, 50]
    peers_list = [2, 5, 10] # Connectivity degree (approx)
    epochs_list = [1, 5, 10] # Communication frequency
    
    configs = list(itertools.product(nodes_list, peers_list, epochs_list))
    
    summary = []
    
    for nodes, peers, epochs in configs:
        run_name = f"scale_n{nodes}_p{peers}_e{epochs}"
        print(f"Running config: {run_name}")
        
        # Determine connectivity prob for random graph to match avg degree ~ peers
        # Avg degree = p * (N-1) => p = peers / (N-1)
        prob = peers / (nodes - 1) if nodes > 1 else 0
        prob = min(1.0, max(0.0, prob))
        
        sim = FederatedSimulation(
            num_nodes=nodes,
            dataset_name='synthetic',
            mode='decentralized',
            connectivity='random',
            connectivity_prob=prob
        )
        
        history = sim.run_rounds(rounds=5, epochs_per_round=epochs)
        
        # Save metrics
        run_dir = os.path.join(output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        pd.DataFrame(history).to_csv(os.path.join(run_dir, 'metrics.csv'), index=False)
        
        # Collect final metrics
        final = history[-1]
        summary.append({
            'nodes': nodes,
            'peers': peers,
            'epochs': epochs,
            'f1': final['f1'],
            'precision': final['precision'],
            'recall': final['recall'],
            'train_loss': final['train_loss'],
            'divergence': final.get('divergence', 0.0),
            'overhead_kb': final.get('overhead_kb', 0.0),
            'round_time_s': final.get('round_time_s', 0.0)
        })
        
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'scalability_summary.csv'), index=False)
    
    # Plot heatmap
    try:
        plot_scalability_heatmap(summary_df, output_dir)
    except Exception as e:
        print(f"Error plotting heatmap: {e}")
        
    print("Scalability study completed.")

if __name__ == "__main__":
    run_scalability_study()
