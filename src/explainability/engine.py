import torch
import torch.nn as nn
import numpy as np
import shap

class ExplainabilityEngine:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def get_gradient_attribution(self, x_input: torch.Tensor) -> np.ndarray:
        """
        Computes input * gradient of the reconstruction loss w.r.t input.
        x_input: (1, seq_len, features)
        """
        x = x_input.clone().detach().to(self.device).requires_grad_(True)
        
        # Forward pass
        x_recon = self.model(x)
        
        # Compute MSE loss for this sample
        loss = nn.MSELoss()(x_recon, x)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Saliency map: |x * grad|
        # gradients: (1, seq_len, features)
        gradients = x.grad.data
        attribution = torch.abs(x * gradients)
        
        return attribution.cpu().numpy()

    def get_shap_values(self, x_background: np.ndarray, x_test: np.ndarray, n_samples: int = 50) -> np.ndarray:
        """
        Computes SHAP values using KernelExplainer (model agnostic but slow) or GradientExplainer.
        Since we want to explain the reconstruction error, we need a wrapper.
        
        x_background: (N, seq_len, features) - summary background dataset
        x_test: (1, seq_len, features) - instance to explain
        """
        
        # We define a prediction function that returns the MSE loss per sample
        def model_loss_wrapper(data_numpy):
            data_tensor = torch.from_numpy(data_numpy).float().to(self.device)
            with torch.no_grad():
                recon = self.model(data_tensor)
                # Compute MSE per sample, summed over time and features for scalar output?
                # SHAP expects (n_samples, output_dim). 
                # If we want to explain "Anomaly Score", output is (n_samples, 1).
                mse = torch.mean((data_tensor - recon) ** 2, dim=(1, 2))
                return mse.cpu().numpy().reshape(-1, 1)

        # Use KernelExplainer for flexibility, though GradientExplainer is faster for Torch
        # But GradientExplainer for Loss function is tricky without custom backward hooks.
        # Let's stick to a simplified approach or just GradientExplainer on the output if it was a classifier.
        # For Autoencoder anomaly detection, we want to explain the Loss.
        
        explainer = shap.KernelExplainer(model_loss_wrapper, x_background)
        shap_values = explainer.shap_values(x_test, nsamples=n_samples)
        
        return np.array(shap_values)

