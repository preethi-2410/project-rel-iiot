import torch
import numpy as np
import networkx as nx
from typing import List, Dict
import copy
from tqdm import tqdm

from src.data.loader import DataManager
from src.federated.node import EdgeNode
from src.federated.peer_exchange import PeerExchange
from src.evaluation.metrics import MetricsCalculator

class FederatedSimulation:
    def __init__(self, 
                 num_nodes: int = 5, 
                 dataset_name: str = 'synthetic',
                 connectivity: str = 'ring', # ring, full, random
                 connectivity_prob: float = 0.5, # for random
                 mode: str = 'decentralized', # decentralized, centralized, local
                 device: str = 'cpu',
                 participation_rate: float = 1.0,
                 drop_rate: float = 0.0,
                 use_dp: bool = False,
                 dp_epsilon: float = 1.0,
                 use_defense: bool = False):
        
        self.num_nodes = num_nodes
        self.mode = mode
        self.device = device
        self.participation_rate = participation_rate
        self.network = PeerExchange(drop_rate=drop_rate)
        
        self.nodes: List[EdgeNode] = []
        self.adjacency_matrix = None
        
        # Load Data
        self.data_manager = DataManager(dataset_name)
        X_train, X_test, y_test, y_train = self.data_manager.load_data()
        
        # Distribute Data (Simple random split for now, can be improved for non-IID)
        # Split train data among nodes
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        train_shards = np.array_split(indices, num_nodes)
        
        # Split test data among nodes (or keep global for avg eval? 
        # Usually edge nodes evaluate on their own local test set).
        test_indices = np.arange(len(X_test))
        np.random.shuffle(test_indices)
        test_shards = np.array_split(test_indices, num_nodes)
        
        input_dim = X_train.shape[2]
        
        for i in range(num_nodes):
            node = EdgeNode(node_id=i, input_dim=input_dim, device=device,
                            use_dp=use_dp, dp_epsilon=dp_epsilon, use_defense=use_defense,
                            seq_len=self.data_manager.window_size, stride=self.data_manager.stride)
            
            node_train_x = X_train[train_shards[i]]
            node_test_x = X_test[test_shards[i]]
            node_test_y = y_test[test_shards[i]]
            
            node.set_data(node_train_x, node_test_x, node_test_y)
            self.nodes.append(node)
            
        if self.mode == 'decentralized':
            self._setup_connectivity(connectivity, connectivity_prob)
        elif self.mode == 'local':
            # No connections
            self.adjacency_matrix = np.zeros((num_nodes, num_nodes))
        elif self.mode == 'centralized':
            # Adjacency not used in the same way, but let's initialize it
            self.adjacency_matrix = np.zeros((num_nodes, num_nodes))
        
    def _setup_connectivity(self, connectivity, prob):
        if connectivity == 'full':
            G = nx.complete_graph(self.num_nodes)
        elif connectivity == 'ring':
            G = nx.cycle_graph(self.num_nodes)
        elif connectivity == 'random':
            G = nx.erdos_renyi_graph(self.num_nodes, prob)
            # Ensure connected
            if self.num_nodes > 1:
                while not nx.is_connected(G):
                     G = nx.erdos_renyi_graph(self.num_nodes, prob)
        else:
            raise ValueError(f"Unknown connectivity: {connectivity}")
            
        self.adjacency_matrix = nx.to_numpy_array(G)
        
    def run_rounds(self, rounds: int = 10, epochs_per_round: int = 1, callback=None):
        history = []
        
        for r in tqdm(range(rounds), desc=f"Federated Rounds ({self.mode})"):
            round_metrics = {'round': r}
            import time
            t_start = time.time()
            
            # Determine active nodes
            num_active = max(1, int(self.num_nodes * self.participation_rate))
            active_indices = np.random.choice(self.num_nodes, num_active, replace=False)
            
            # 1. Local Training
            losses = []
            for i in active_indices:
                node = self.nodes[i]
                loss = node.train_local(epochs=epochs_per_round)
                losses.append(loss)
            
            # 2. Communication & Aggregation
            # Get params from active nodes
            active_params_map = {i: self.nodes[i].get_parameters() for i in active_indices}
            
            if self.mode == 'decentralized':
                for i in active_indices:
                    node = self.nodes[i]
                    neighbors = np.where(self.adjacency_matrix[i] == 1)[0]
                    
                    # Only receive from active neighbors
                    active_neighbors = [n for n in neighbors if n in active_indices]
                    
                    # Simulate Network (Loss/Latency)
                    received_params = []
                    weights = []
                    for n_idx in active_neighbors:
                        param = active_params_map[n_idx]
                        # Simulate sending
                        latency = self.network.sample_latency()
                        received = self.network.send(param)
                        if received is not None:
                            received_params.append(received)
                            weights.append(self.network.staleness_weight(latency))
                            
                    node.aggregate_peers(received_params, weights if len(weights) > 0 else None)
                    
            elif self.mode == 'centralized':
                # Average all active params
                if len(active_params_map) > 0:
                    keys = list(active_params_map.keys())
                    avg_params = copy.deepcopy(active_params_map[keys[0]])
                    
                    for key in avg_params.keys():
                        for i in keys[1:]:
                            avg_params[key] += active_params_map[i][key]
                        avg_params[key] = torch.div(avg_params[key], len(keys))
                    
                    # Broadcast back
                    for node in self.nodes:
                        node.set_parameters(avg_params)
            
            # For 'local', do nothing
            
            # Update thresholds
            for node in self.nodes:
                node.update_threshold()

                
            # 3. Evaluation
            node_metrics_list = []
            for node in self.nodes:
                m = node.evaluate()
                node_metrics_list.append(m)
                
            # Aggregate metrics
            avg_f1 = np.mean([m['f1'] for m in node_metrics_list])
            avg_prec = np.mean([m['precision'] for m in node_metrics_list])
            avg_rec = np.mean([m['recall'] for m in node_metrics_list])
            avg_latency = np.mean([m.get('latency', 0.0) for m in node_metrics_list])
            
            round_metrics['f1'] = avg_f1
            round_metrics['precision'] = avg_prec
            round_metrics['recall'] = avg_rec
            round_metrics['train_loss'] = np.mean(losses)
            round_metrics['latency'] = avg_latency
            round_metrics['overhead_kb'] = self.get_communication_overhead()
            model_states = [n.model.state_dict() for n in self.nodes]
            round_metrics['divergence'] = MetricsCalculator.compute_model_divergence(model_states)
            
            history.append(round_metrics)

            if callback:
                callback(self, r)
            round_metrics['round_time_s'] = time.time() - t_start
            
        return history

    def get_communication_overhead(self):
        # Estimate KB exchanged per round
        # Model size * avg_degree * num_nodes
        param_size = 0
        for p in self.nodes[0].model.parameters():
            param_size += p.nelement() * 4 # 4 bytes float32
        
        avg_degree = np.mean(np.sum(self.adjacency_matrix, axis=1))
        
        # Each node sends its model to all neighbors (or neighbors pull, same diff)
        total_bytes = param_size * avg_degree * self.num_nodes
        return total_bytes / 1024.0 # KB
