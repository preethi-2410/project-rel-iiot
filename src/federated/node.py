import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from typing import List, Dict, Tuple

from src.models.lstm_ae import LSTMAutoencoder
from src.explainability.engine import ExplainabilityEngine
from src.utils.metrics import get_threshold, calculate_metrics

class EdgeNode:
    def __init__(self, node_id: int, input_dim: int, device: str = 'cpu'):
        self.node_id = node_id
        self.device = device
        self.model = LSTMAutoencoder(input_dim=input_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        
        self.train_loader = None
        self.test_data = None
        self.test_labels = None
        
        self.threshold = None
        self.explainer = ExplainabilityEngine(self.model, device)
        
    def set_data(self, X_train, X_test, y_test):
        # X_train: healthy data for training
        tensor_x = torch.Tensor(X_train)
        dataset = TensorDataset(tensor_x, tensor_x) # Autoencoder target is input
        self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.test_data = X_test
        self.test_labels = y_test
        
        # Initialize threshold based on initial untrained model (or expected behavior)
        # Better to initialize after first training round or pre-training.
        
    def train_local(self, epochs: int = 1):
        self.model.train()
        epoch_losses = []
        for epoch in range(epochs):
            batch_losses = []
            for batch_x, target in self.train_loader:
                batch_x = batch_x.to(self.device)
                target = target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                batch_losses.append(loss.item())
            epoch_losses.append(np.mean(batch_losses))
        return np.mean(epoch_losses)

    def update_threshold(self):
        # Calculate reconstruction error on training data (healthy)
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, target in self.train_loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)
                # Loss per sample
                loss = torch.mean((output - batch_x) ** 2, dim=(1, 2))
                losses.extend(loss.cpu().numpy())
        
        self.threshold = get_threshold(np.array(losses), k=3.0)
        return self.threshold

    def get_parameters(self) -> Dict:
        return copy.deepcopy(self.model.state_dict())

    def set_parameters(self, parameters: Dict):
        self.model.load_state_dict(parameters)

    def aggregate_peers(self, peer_params_list: List[Dict], weights: List[float] = None):
        """
        Federated Averaging (FedAvg) adaptation for P2P.
        Averages neighbor models with current model.
        """
        if not peer_params_list:
            return

        current_params = self.model.state_dict()
        new_params = copy.deepcopy(current_params)
        
        if weights is None:
            # Simple average including self
            total_peers = len(peer_params_list) + 1
            weight = 1.0 / total_peers
            
            for key in new_params.keys():
                new_params[key] = new_params[key] * weight # self
                for peer_param in peer_params_list:
                    new_params[key] += peer_param[key] * weight
        else:
            # Weighted average
            # Ensure weights sum to 1 (including self weight if handled externally)
            # For simplicity here, let's assume equal weights for all peers + self
            pass # TODO: implement explicit weighted avg if needed
            
        self.model.load_state_dict(new_params)

    def evaluate(self):
        self.model.eval()
        tensor_test = torch.Tensor(self.test_data).to(self.device)
        
        with torch.no_grad():
            reconstruction = self.model(tensor_test)
            # MSE per sample
            loss = torch.mean((reconstruction - tensor_test) ** 2, dim=(1, 2)).cpu().numpy()
            
        preds = (loss > self.threshold).astype(int)
        metrics = calculate_metrics(self.test_labels, preds)
        metrics['avg_loss'] = np.mean(loss)
        return metrics

    def explain_sample(self, sample_idx: int):
        """
        Explain a specific sample from test set.
        """
        x = torch.Tensor(self.test_data[sample_idx:sample_idx+1])
        attribution = self.explainer.get_gradient_attribution(x)
        return attribution
