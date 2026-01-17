import yaml
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Run Rel-IIoT Experiment")
    parser.add_argument('--config', type=str, help="Path to YAML configuration file")
    parser.add_argument('--output', type=str, default='results', help="Output directory")
    args, unknown = parser.parse_known_args()
    return args
