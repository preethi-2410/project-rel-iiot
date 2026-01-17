import numpy as np
import time
from typing import List, Callable

class ExplainabilityEvaluator:
    @staticmethod
    def calculate_stability(explanations: List[np.ndarray]) -> float:
        """
        Calculates stability of explanations over time (e.g., across rounds).
        Uses Jaccard similarity of top-k features or Cosine similarity.
        """
        if len(explanations) < 2:
            return 1.0
            
        similarities = []
        for i in range(len(explanations) - 1):
            e1 = explanations[i].flatten()
            e2 = explanations[i+1].flatten()
            
            # Cosine similarity
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)
            
            if norm1 == 0 or norm2 == 0:
                sim = 0.0
            else:
                sim = np.dot(e1, e2) / (norm1 * norm2)
            similarities.append(sim)
            
        return float(np.mean(similarities))

    @staticmethod
    def measure_runtime(func: Callable, *args, **kwargs) -> float:
        """
        Measures execution time of an explainability function in seconds.
        """
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        return end - start

    @staticmethod
    def attribution_overlap(explanation: np.ndarray, fault_indices: np.ndarray, top_k: int = 5) -> float:
        flat = np.abs(explanation).mean(axis=0)
        if flat.ndim > 1:
            flat = flat.mean(axis=0)
        order = np.argsort(-flat)
        top = set(order[:top_k])
        faults = set(fault_indices.tolist())
        inter = len(top.intersection(faults))
        return float(inter) / float(top_k)
