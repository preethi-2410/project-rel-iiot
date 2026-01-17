import os
import pandas as pd
from typing import Tuple

class SecomLoader:
    def __init__(self, data_dir: str = 'data/secom'):
        self.data_dir = data_dir
        
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data_file = os.path.join(self.data_dir, 'secom.data')
        labels_file = os.path.join(self.data_dir, 'secom_labels.data')
        
        if not os.path.exists(data_file):
             raise FileNotFoundError(f"SECOM file not found: {data_file}")
             
        # SECOM is space separated, NaN is represented often by empty or specific val? 
        # Usually it's standard CSV/space.
        data = pd.read_csv(data_file, sep='\\s+', header=None)
        labels = pd.read_csv(labels_file, sep='\\s+', header=None)
        
        # Preprocessing: Handle NaNs (fill with mean or 0)
        data = data.fillna(0)
        
        return data, labels
