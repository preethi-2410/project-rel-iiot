import torch
import shap
import numpy as np

class ExplainabilityEngine:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.explainer = None
        
    def prepare_explainer(self, background_data):
        # background_data: (N, seq_len, features)
        # Convert to torch
        data_tensor = torch.FloatTensor(background_data).to(self.device)
        
        # Limit background size
        if len(data_tensor) > 100:
            indices = np.random.choice(len(data_tensor), 100, replace=False)
            data_tensor = data_tensor[indices]
            
        # Use GradientExplainer
        try:
            self.explainer = shap.GradientExplainer(self.model, data_tensor)
        except Exception as e:
            print(f"Warning: SHAP init failed: {e}")
            self.explainer = None

    def get_gradient_attribution(self, x):
        x = x.to(self.device)
        x.requires_grad = True
        out = self.model(x)
        grad = torch.autograd.grad(outputs=out.sum(), inputs=x, retain_graph=False)[0]
        return grad.detach().cpu().numpy()

    def get_shap_attribution(self, x):
        if self.explainer is None:
            return None
        vals = self.explainer.shap_values(x)
        if isinstance(vals, list):
            vals = vals[0]
        return vals
        
    def explain(self, instance):
        # instance: (seq_len, features) or (1, seq_len, features)
        if self.explainer is None:
            return None
            
        if isinstance(instance, np.ndarray):
            if len(instance.shape) == 2:
                instance = instance[np.newaxis, ...]
            instance_tensor = torch.FloatTensor(instance).to(self.device)
        else:
            instance_tensor = instance
            
        if not instance_tensor.requires_grad:
            instance_tensor.requires_grad = True
        
        shap_values = self.explainer.shap_values(instance_tensor)
        
        return shap_values

    def get_gradient_attribution(self, x):
        """
        Computes gradient-based attribution (Saliency Map).
        Returns: (seq_len, features) numpy array
        """
        self.model.eval()
        if not x.requires_grad:
            x.requires_grad = True
        
        output = self.model(x)
        # Loss: reconstruction error (MSE)
        loss = torch.mean((output - x) ** 2)
        
        self.model.zero_grad()
        loss.backward()
        
        # Saliency: |gradient|
        if x.grad is not None:
            saliency = x.grad.abs().detach().cpu().numpy()
            return saliency
        return np.zeros_like(x.detach().cpu().numpy())
