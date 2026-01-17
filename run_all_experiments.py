import os
import sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from run_baselines import main as run_baselines_main
from run_scalability import main as run_scalability_main
from run_explainability import main as run_explainability_main

def set_seeds(seed=42):
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass

def main():
    set_seeds(42)
    os.makedirs('results', exist_ok=True)
    run_baselines_main()
    run_scalability_main()
    run_explainability_main()

if __name__ == '__main__':
    main()
