import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.federated.system import FederatedSimulation
from src.explainability.eval_explainability import ExplainabilityEvaluator

def run_shap_study(output_dir='results/explainability_shap', dataset='synthetic', rounds=5):
    os.makedirs(output_dir, exist_ok=True)
    sim = FederatedSimulation(num_nodes=3, dataset_name=dataset, mode='decentralized', connectivity='ring')

    node_idx = 0
    sample_idx = 0
    explanations = []
    runtimes = []

    def on_round_end(simulation, round_num):
        node = simulation.nodes[node_idx]
        if node.train_loader is None or node.test_data is None:
            return
        bg = node.train_loader.dataset.tensors[0].cpu().numpy()
        node.explainer.prepare_explainer(bg)
        x = node.test_data[sample_idx:sample_idx+1]
        rt = ExplainabilityEvaluator.measure_runtime(node.explainer.get_shap_attribution, x)
        explanations.append(node.explainer.get_shap_attribution(x))
        runtimes.append(rt)

    sim.run_rounds(rounds=rounds, epochs_per_round=1, callback=on_round_end)

    stability = ExplainabilityEvaluator.calculate_stability([e if e is not None else np.zeros((1,)) for e in explanations])
    avg_runtime = float(np.mean(runtimes)) if len(runtimes) > 0 else 0.0

    df = pd.DataFrame({'metric': ['stability', 'avg_runtime'], 'value': [stability, avg_runtime]})
    df.to_csv(os.path.join(output_dir, 'shap_metrics.csv'), index=False)

if __name__ == '__main__':
    run_shap_study()
