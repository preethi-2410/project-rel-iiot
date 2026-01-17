import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Union
from sklearn.preprocessing import MinMaxScaler
from .synthetic import SyntheticGenerator
import os

class DataManager:
    def __init__(self, dataset_name: str, window_size: int = 50, stride: int = 1, seed: int = 42):
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.stride = stride
        self.seed = seed
        self.scaler = MinMaxScaler()
        self.rng = np.random.default_rng(seed)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads data and splits into train/test.
        Train data contains ONLY healthy sequences.
        Test data contains full lifecycle (healthy -> fault).
        
        Returns:
            train_windows: (n_train_samples, window_size, n_features)
            test_windows: (n_test_samples, window_size, n_features)
            test_labels: (n_test_samples,) 0=healthy, 1=anomaly
            train_labels: (n_train_samples,) all 0s
        """
        if self.dataset_name == 'synthetic':
            return self._load_synthetic()
        elif self.dataset_name == 'cmapss':
            return self._load_cmapss()
        elif self.dataset_name == 'secom':
            return self._load_secom()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _create_windows(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        for i in range(0, len(data) - self.window_size, self.stride):
            X.append(data[i : i + self.window_size])
            # Label is 1 if any point in the window is anomalous? 
            # Or just the last point? Usually last point or majority. 
            # Let's use the last point for detection latency evaluation.
            y.append(labels[i + self.window_size - 1]) 
        return np.array(X), np.array(y)

    def _load_synthetic(self):
        gen = SyntheticGenerator(seed=self.seed)
        # Generate enough samples
        # Train: 20 healthy runs (cut before degradation)
        # Test: 10 full runs
        
        raw_train_data = []
        
        # Generate pure healthy data for training
        for _ in range(50):
            data, labels = gen.generate_series(length=500, degradation_start=500) # degradation starts at end -> all healthy
            raw_train_data.append(data)
            
        raw_test_data = []
        raw_test_labels = []
        for _ in range(20):
            data, labels = gen.generate_series(length=1000, degradation_start=500)
            raw_test_data.append(data)
            raw_test_labels.append(labels)
            
        # Fit scaler on concatenated train data
        flat_train = np.concatenate(raw_train_data, axis=0)
        self.scaler.fit(flat_train)
        
        # Transform and window train data
        train_windows_list = []
        train_labels_list = []
        for d in raw_train_data:
            d_scaled = self.scaler.transform(d)
            # All labels are 0 for train
            l = np.zeros(len(d))
            w, l_w = self._create_windows(d_scaled, l)
            train_windows_list.append(w)
            train_labels_list.append(l_w)
            
        # Transform and window test data
        test_windows_list = []
        test_labels_list = []
        for d, l in zip(raw_test_data, raw_test_labels):
            d_scaled = self.scaler.transform(d)
            w, l_w = self._create_windows(d_scaled, l)
            test_windows_list.append(w)
            test_labels_list.append(l_w)
            
        X_train = np.concatenate(train_windows_list, axis=0)
        y_train = np.concatenate(train_labels_list, axis=0)
        X_test = np.concatenate(test_windows_list, axis=0)
        y_test = np.concatenate(test_labels_list, axis=0)
        
        return X_train, X_test, y_test, y_train

    def _load_cmapss(self):
        # Placeholder for C-MAPSS
        # In a real scenario, this would load from files
        # For now, fallback to synthetic or raise warning
        print("Warning: C-MAPSS data not found. Returning synthetic data.")
        return self._load_synthetic()

    def _load_secom(self):
        # Placeholder for SECOM
        print("Warning: SECOM data not found. Returning synthetic data.")
        return self._load_synthetic()
