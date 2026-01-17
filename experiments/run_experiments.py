import torch
import sys
import os
import pandas as pd
import numpy as np
import argparse

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.federated.system import FederatedSimulation
from src.utils.plotting import plot_convergence, plot_scalability, plot_communication_overhead
from src.utils.config import load_config, parse_args

def run_from_config(config):
    print("Running Experiment from Config...")
    
    # Extract params
    fed = config.get('federated', {})
    net = config.get('network', {})
    sec = config.get('security', {})
    data = config.get('data', {})
    
    output_dir = config.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    sim = FederatedSimulation(
        num_nodes=fed.get('num_nodes', 5),
        dataset_name=data.get('dataset_name', 'synthetic'),
        connectivity=fed.get('connectivity', 'ring'),
        connectivity_prob=fed.get('connectivity_prob', 0.5),
        mode=fed.get('mode', 'decentralized'),
        device=config.get('device', 'cpu'),
        participation_rate=net.get('participation_rate', 1.0),
        drop_rate=net.get('drop_rate', 0.0),
        use_dp=sec.get('use_dp', False),
        dp_epsilon=sec.get('dp_epsilon', 1.0),
        use_defense=sec.get('use_defense', False)
    )
    
    history = sim.run_rounds(
        rounds=fed.get('num_rounds', 10),
        epochs_per_round=fed.get('epochs_per_round', 1)
    )
    
    # Save results
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(output_dir, 'config_experiment_results.csv'), index=False)
    print(f"Experiment completed. Results in {output_dir}")


def run_baseline_comparison(output_dir):
    print("Running Baseline Comparison (Decentralized vs Centralized vs Local)...")
    
    modes = ['decentralized', 'centralized', 'local']
    all_history = []
    
    for mode in modes:
        print(f"  Mode: {mode}")
        sim = FederatedSimulation(num_nodes=5, dataset_name='synthetic', mode=mode)
        history = sim.run_rounds(rounds=10, epochs_per_round=1)
        
        for h in history:
            h['method'] = mode
            all_history.append(h)
            
    df = pd.DataFrame(all_history)
    df.to_csv(os.path.join(output_dir, 'baseline_results.csv'), index=False)
    plot_convergence(df, output_dir)
    print("Baseline comparison done.")

def run_scalability_study(output_dir):
    print("Running Scalability Study (Nodes: 5, 10, 20)...")
    
    node_counts = [5, 10, 20]
    results = []
    
    for n in node_counts:
        print(f"  Nodes: {n}")
        sim = FederatedSimulation(num_nodes=n, dataset_name='synthetic', mode='decentralized', connectivity='ring')
        history = sim.run_rounds(rounds=5, epochs_per_round=1) # Reduced rounds for speed
        
        final_f1 = history[-1]['f1']
        results.append({
            'num_nodes': n,
            'f1': final_f1,
            'method': 'Rel-IIoT (Ring)'
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'scalability_results.csv'), index=False)
    plot_scalability(df, output_dir)
    print("Scalability study done.")

def main():
    args = parse_args()
    
    if args.config:
        config = load_config(args.config)
        # Override output dir if provided in args
        if args.output != 'results':
            config['output_dir'] = args.output
        run_from_config(config)
    else:
        os.makedirs(args.output, exist_ok=True)
        run_baseline_comparison(args.output)
        run_scalability_study(args.output)
        print(f"All experiments completed. Results in {args.output}")


if __name__ == "__main__":
    main()
