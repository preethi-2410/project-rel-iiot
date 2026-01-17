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
        from .cmapss import CMapssLoader
        try:
            loader = CMapssLoader()
            train_df, test_df, rul_df = loader.load()
        except Exception:
            return self._load_synthetic()

        feature_cols = [c for c in train_df.columns if c.startswith('setting_') or c.startswith('s_')]

        train_windows_list = []
        train_labels_list = []
        test_windows_list = []
        test_labels_list = []

        units = train_df['unit_nr'].unique()
        flat_train = []
        for u in units:
            seq = train_df[train_df['unit_nr'] == u].sort_values('time_cycles')
            data_u = seq[feature_cols].values
            max_len = len(data_u)
            healthy_end = int(max_len * 0.7)
            flat_train.append(data_u[:healthy_end])

        if len(flat_train) == 0:
            return self._load_synthetic()

        self.scaler.fit(np.concatenate(flat_train, axis=0))

        for u in units:
            seq = train_df[train_df['unit_nr'] == u].sort_values('time_cycles')
            data_u = seq[feature_cols].values
            max_len = len(data_u)
            healthy_end = int(max_len * 0.7)
            d_scaled = self.scaler.transform(data_u[:healthy_end])
            l = np.zeros(len(d_scaled))
            w, l_w = self._create_windows(d_scaled, l)
            if len(w) > 0:
                train_windows_list.append(w)
                train_labels_list.append(l_w)

        test_units = test_df['unit_nr'].unique()
        for idx, u in enumerate(test_units):
            seq = test_df[test_df['unit_nr'] == u].sort_values('time_cycles')
            data_u = seq[feature_cols].values
            d_scaled = self.scaler.transform(data_u)
            max_len = len(d_scaled)
            anomaly_start = int(max_len * 0.8)
            labels_u = np.zeros(max_len)
            if anomaly_start < max_len:
                labels_u[anomaly_start:] = 1
            w, l_w = self._create_windows(d_scaled, labels_u)
            if len(w) > 0:
                test_windows_list.append(w)
                test_labels_list.append(l_w)

        X_train = np.concatenate(train_windows_list, axis=0) if len(train_windows_list) > 0 else np.empty((0, self.window_size, len(feature_cols)))
        y_train = np.concatenate(train_labels_list, axis=0) if len(train_labels_list) > 0 else np.empty((0,))
        X_test = np.concatenate(test_windows_list, axis=0) if len(test_windows_list) > 0 else np.empty((0, self.window_size, len(feature_cols)))
        y_test = np.concatenate(test_labels_list, axis=0) if len(test_labels_list) > 0 else np.empty((0,))

        return X_train, X_test, y_test, y_train

    def _load_secom(self):
        from .secom import SecomLoader
        try:
            loader = SecomLoader()
            data_df, labels_df = loader.load()
        except Exception:
            return self._load_synthetic()

        labels = labels_df.iloc[:, 0].values
        labels = np.where(labels == -1, 0, 1)

        data = data_df.values

        healthy_mask = labels == 0
        train_data = data[healthy_mask]

        if train_data.shape[0] == 0:
            return self._load_synthetic()

        self.scaler.fit(train_data)

        d_train = self.scaler.transform(train_data)
        l_train = np.zeros(d_train.shape[0])

        X_train, y_train = self._create_windows(d_train, l_train)

        d_test = self.scaler.transform(data)
        X_test, y_test = self._create_windows(d_test, labels[:d_test.shape[0]])

        return X_train, X_test, y_test, y_train
