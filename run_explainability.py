import os
import sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from experiments.explainability_driver import run_explainability_study
from src.explainability.shap_runner import run_shap_study

def set_seeds(seed=42):
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass

def main():
    set_seeds(42)
    run_explainability_study()
    run_shap_study()

if __name__ == '__main__':
    main()
