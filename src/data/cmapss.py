import os
import pandas as pd
from typing import Tuple

class CMapssLoader:
    def __init__(self, data_dir: str = 'data/cmapss', sub_dataset: str = 'FD001'):
        self.data_dir = data_dir
        self.sub_dataset = sub_dataset
        self.index_names = ['unit_nr', 'time_cycles']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names = [f's_{i}' for i in range(1, 22)]
        self.col_names = self.index_names + self.setting_names + self.sensor_names
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_file = os.path.join(self.data_dir, f'train_{self.sub_dataset}.txt')
        test_file = os.path.join(self.data_dir, f'test_{self.sub_dataset}.txt')
        rul_file = os.path.join(self.data_dir, f'RUL_{self.sub_dataset}.txt')
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"C-MAPSS file not found: {train_file}")
            
        train = pd.read_csv(train_file, sep='\\s+', header=None, names=self.col_names)
        test = pd.read_csv(test_file, sep='\\s+', header=None, names=self.col_names)
        rul = pd.read_csv(rul_file, sep='\\s+', header=None, names=['RUL'])
        return train, test, rul
