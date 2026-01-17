import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def get_threshold(losses: np.ndarray, k: float = 3.0) -> float:
    mean = np.mean(losses)
    std = np.std(losses)
    return mean + k * std

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
