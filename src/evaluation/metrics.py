import numpy as np
from typing import List, Dict
import torch

class MetricsCalculator:
    @staticmethod
    def compute_detection_latency(y_true: np.ndarray, y_pred: np.ndarray, stride: int = 1) -> float:
        """
        Computes average time (in time steps) between anomaly start and first detection.
        """
        # Identify continuous anomaly segments in y_true
        # y_true is 0s and 1s.
        # Find rising edges
        true_starts = np.where(np.diff(y_true, prepend=0) == 1)[0]
        
        latencies = []
        for start in true_starts:
            # Look forward from start
            # End of this anomaly segment?
            # Simplified: just look until next 0 or end
            future = y_pred[start:]
            det_indices = np.where(future == 1)[0]
            if len(det_indices) > 0:
                latencies.append(det_indices[0] * stride)
            else:
                # Missed detection? Penalize? Or ignore for latency calc (metrics usually cover recall)
                pass
                
        return float(np.mean(latencies)) if latencies else 0.0
    
    @staticmethod
    def compute_model_divergence(models: List[Dict]) -> float:
        """
        Computes Euclidean distance of model parameters from their centroid.
        """
        if len(models) < 2:
            return 0.0
            
        flat_params = []
        for state_dict in models:
            # Flatten all params
            p_list = []
            for k in sorted(state_dict.keys()):
                p_list.append(state_dict[k].cpu().numpy().flatten())
            flat_params.append(np.concatenate(p_list))
            
        flat_params = np.array(flat_params)
        centroid = np.mean(flat_params, axis=0)
        
        distances = np.linalg.norm(flat_params - centroid, axis=1)
        return float(np.mean(distances))
