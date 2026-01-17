import os
import sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from experiments.scalability_driver import run_scalability_study

def set_seeds(seed=42):
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass

def main():
    set_seeds(42)
    out_dir = 'results/scalability'
    run_scalability_study(output_dir=out_dir)

if __name__ == '__main__':
    main()
