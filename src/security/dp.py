import numpy as np
import torch
from typing import Dict

class DifferentialPrivacy:
    def __init__(self, epsilon: float = 1.0, sensitivity: float = 0.1):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        
    def add_noise(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Adds Laplace noise to parameters.
        """
        noisy_params = {}
        scale = self.sensitivity / self.epsilon
        
        for name, param in parameters.items():
            noise = np.random.laplace(0, scale, param.shape)
            noisy_params[name] = param + torch.from_numpy(noise).float().to(param.device)
            
        return noisy_params
