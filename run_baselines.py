import os
import sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.baselines.local_only import run_local_baseline
from src.baselines.centralized_fl import run_centralized_baseline

def set_seeds(seed=42):
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass

def main(dataset='synthetic'):
    set_seeds(42)
    out_dir = 'results/baselines'
    os.makedirs(out_dir, exist_ok=True)
    run_local_baseline(output_dir=out_dir, dataset=dataset)
    run_centralized_baseline(output_dir=out_dir, dataset=dataset)

if __name__ == '__main__':
    main()
