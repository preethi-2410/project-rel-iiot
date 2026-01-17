import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

def plot_convergence(history_df: pd.DataFrame, save_path: str):
    """
    Plots metric (e.g., F1) over rounds for different configurations.
    history_df should have columns: round, value, method (or config)
    """
    set_style()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=history_df, x='round', y='f1', hue='method', style='method', markers=True)
    plt.title('Convergence Analysis')
    plt.xlabel('Federated Round')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'convergence.png'))
    plt.close()

def plot_communication_overhead(data: pd.DataFrame, save_path: str):
    """
    Plots overhead vs frequency or other param.
    """
    set_style()
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x='frequency', y='overhead_kb', hue='method')
    plt.title('Communication Overhead')
    plt.xlabel('Communication Frequency')
    plt.ylabel('Overhead per Round (KB)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'overhead.png'))
    plt.close()

def plot_scalability(data: pd.DataFrame, save_path: str):
    """
    Metric vs Node Count
    """
    set_style()
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=data, x='num_nodes', y='f1', hue='method', marker='o')
    plt.title('Scalability: Performance vs Node Count')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Final F1 Score')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'scalability.png'))
    plt.close()

def plot_explainability_stability(stability_scores: List[float], save_path: str):
    set_style()
    plt.figure(figsize=(8, 5))
    plt.plot(stability_scores, marker='o')
    plt.title('Explainability Stability over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Stability Score (Jaccard/Cosine)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'explainability_stability.png'))
    plt.close()
