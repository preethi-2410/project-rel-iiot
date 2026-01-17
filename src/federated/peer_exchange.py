import numpy as np
import time
import asyncio
from typing import List, Dict, Any

class PeerExchange:
    def __init__(self, 
                 latency_mean: float = 0.0, 
                 latency_std: float = 0.0, 
                 drop_rate: float = 0.0,
                 staleness_alpha: float = 1.0):
        """
        Simulates network conditions.
        latency_mean: mean latency in seconds (simulated sleep or cost)
        drop_rate: probability of message loss (0.0 to 1.0)
        """
        self.latency_mean = latency_mean
        self.latency_std = latency_std
        self.drop_rate = drop_rate
        self.staleness_alpha = staleness_alpha
        self.rng = np.random.default_rng()

    def send(self, message: Any) -> Any:
        """
        Simulates sending a message. Returns None if dropped.
        In a real async system, this would await.
        Here we return the message if successful, or None.
        """
        # Simulate Drop
        if self.rng.random() < self.drop_rate:
            return None
            
        # Simulate Latency (just calculation, we don't actually sleep in this synchronous sim usually, 
        # unless we want to simulate wall-clock time)
        latency = max(0, self.rng.normal(self.latency_mean, self.latency_std))
        
        # If we were tracking global time, we would add latency to the timestamp.
        # For this prototype, we just tag it.
        return message

    def sample_latency(self) -> float:
        return max(0.0, self.rng.normal(self.latency_mean, self.latency_std))

    def staleness_weight(self, latency: float) -> float:
        return float(np.exp(-self.staleness_alpha * latency))

    def broadcast(self, message: Any, neighbors: List[int]) -> Dict[int, Any]:
        """
        Sends to multiple neighbors.
        Returns map of {neighbor_id: received_message} (excluding dropped)
        """
        results = {}
        for n in neighbors:
            res = self.send(message)
            if res is not None:
                results[n] = res
        return results
