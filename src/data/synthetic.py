import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

class SyntheticGenerator:
    """
    Generates synthetic IIoT sensor data simulating machine health.
    States: Healthy -> Degrading -> Faulty
    """
    def __init__(self, n_sensors: int = 5, noise_level: float = 0.05, seed: int = 42):
        self.n_sensors = n_sensors
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)
        
    def generate_series(self, length: int = 1000, degradation_start: int = 500, failure_point: int = 900) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a multivariate time series.
        Returns:
            data: (length, n_sensors)
            labels: (length,) 0=healthy, 1=degrading/faulty (for ground truth)
        """
        t = np.linspace(0, 100, length)
        data = np.zeros((length, self.n_sensors))
        
        # Base healthy behavior: Sine waves with different frequencies/phases
        for i in range(self.n_sensors):
            freq = self.rng.uniform(0.1, 0.5)
            phase = self.rng.uniform(0, 2*np.pi)
            data[:, i] = np.sin(freq * t + phase)
            
        # Add degradation (linear drift + variance increase)
        degradation = np.zeros_like(data)
        if degradation_start < length:
            steps = np.arange(length) - degradation_start
            mask = steps > 0
            
            # Exponential degradation trend
            drift = np.exp(steps[mask] * 0.01) - 1
            
            for i in range(self.n_sensors):
                # Randomly choose if a sensor is affected by degradation
                if self.rng.random() > 0.3: 
                    degradation[mask, i] += drift * self.rng.uniform(0.1, 0.5)
                    # Add increasing noise
                    noise_amp = steps[mask] * 0.001
                    degradation[mask, i] += self.rng.normal(0, 1, size=mask.sum()) * noise_amp

        data += degradation
        
        # Add baseline noise
        data += self.rng.normal(0, self.noise_level, size=data.shape)
        
        # Create labels
        labels = np.zeros(length)
        labels[degradation_start:] = 1 # Mark as anomaly from degradation start
        
        return data, labels

    def generate_dataset(self, n_samples: int, length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        dataset = []
        for _ in range(n_samples):
            # Randomize degradation start
            deg_start = int(self.rng.uniform(0.3, 0.7) * length)
            dataset.append(self.generate_series(length, deg_start))
        return dataset
