import os
import pandas as pd
import json

class ExperimentLogger:
    def __init__(self, output_dir, experiment_name):
        self.output_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.logs = []
        
    def log_round(self, round_id, metrics):
        # metrics is a dict
        entry = metrics.copy()
        entry['round'] = round_id
        self.logs.append(entry)
        
    def save(self):
        if not self.logs:
            return
        df = pd.DataFrame(self.logs)
        df.to_csv(os.path.join(self.output_dir, 'metrics.csv'), index=False)
        print(f"Metrics saved to {os.path.join(self.output_dir, 'metrics.csv')}")
