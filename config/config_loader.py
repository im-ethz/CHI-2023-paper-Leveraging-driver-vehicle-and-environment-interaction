import json


def load_config(filepath):
    with open(filepath) as f:
        config_data = json.load(f)
    return config_data
