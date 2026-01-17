import sys
import os
import numpy as np
import pandas as pd
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.federated.system import FederatedSimulation
from src.explainability.eval_explainability import ExplainabilityEvaluator

def run_explainability_study():
    print("=== Explainability Stability Study ===")
    
    # 1. Setup Simulation
    # Use synthetic data for speed
    sim = FederatedSimulation(
        num_nodes=3,
        dataset_name='synthetic',
        mode='decentralized',
        connectivity='ring'
    )
    
    # Track explanations for Node 0, Sample 0
    node_idx = 0
    sample_idx = 0
    
    # Store explanations across rounds
    explanations_history = []
    
    def on_round_end(simulation, round_num):
        node = simulation.nodes[node_idx]
        
        # Ensure we have test data
        if node.test_data is None or len(node.test_data) == 0:
            return

        # Get explanation for the sample
        # We need to measure runtime too
        start = time.time()
        # explain_sample returns numpy array
        explanation = node.explain_sample(sample_idx)
        end = time.time()
        
        runtime = end - start
        
        explanations_history.append({
            'round': round_num,
            'explanation': explanation,
            'runtime': runtime
        })
        
    # 2. Run Simulation
    print("Running simulation with explainability tracking...")
    sim.run_rounds(rounds=10, epochs_per_round=1, callback=on_round_end)
    
    # 3. Calculate Stability
    print("\nCalculating Metrics...")
    if not explanations_history:
        print("No explanations collected!")
        return

    expls = [item['explanation'] for item in explanations_history]
    
    stability_score = ExplainabilityEvaluator.calculate_stability(expls)
    avg_runtime = np.mean([item['runtime'] for item in explanations_history])
    
    print(f"Stability Score (Cosine Sim): {stability_score:.4f}")
    print(f"Avg Explanation Runtime: {avg_runtime:.4f}s")
    
    # 4. Save Results
    results = {
        'metric': ['stability', 'avg_runtime'],
        'value': [stability_score, avg_runtime]
    }
    df = pd.DataFrame(results)
    
    output_dir = 'results/explainability'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Save raw history for detailed analysis
    history_data = []
    for item in explanations_history:
        history_data.append({
            'round': item['round'],
            'runtime': item['runtime'],
            'explanation_norm': np.linalg.norm(item['explanation'])
        })
    pd.DataFrame(history_data).to_csv(os.path.join(output_dir, 'history.csv'), index=False)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    run_explainability_study()
