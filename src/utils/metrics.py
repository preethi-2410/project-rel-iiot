import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def get_threshold(losses: np.ndarray, k: float = 3.0) -> float:
    return np.mean(losses) + k * np.std(losses)
