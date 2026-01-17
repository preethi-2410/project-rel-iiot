import torch
import numpy as np
from typing import List, Dict

class PoisoningDefense:
    def __init__(self, threshold_std: float = 2.0):
        self.threshold_std = threshold_std
        
    def filter_updates(self, current_params: Dict, neighbor_updates: List[Dict]) -> List[Dict]:
        """
        Filters out updates that are too far from the median of the distribution.
        Simple statistical outlier detection.
        """
        if len(neighbor_updates) < 3:
            return neighbor_updates
            
        # We check layer by layer or based on a global norm
        valid_updates = []
        
        # Calculate norms of updates (diff from current)
        norms = []
        for update in neighbor_updates:
            diff_norm = 0.0
            for k in current_params.keys():
                diff = update[k] - current_params[k]
                diff_norm += torch.norm(diff).item()
            norms.append(diff_norm)
            
        norms = np.array(norms)
        median = np.median(norms)
        std = np.std(norms)
        
        # Filter
        threshold = median + self.threshold_std * std
        
        for i, norm in enumerate(norms):
            if norm <= threshold:
                valid_updates.append(neighbor_updates[i])
                
        return valid_updates
