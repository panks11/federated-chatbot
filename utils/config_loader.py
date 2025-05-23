# utils/config_loader.py

import yaml
import os


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_config(base_path="config/base_config.yaml", override_path=None):
    """
    Load base config, optionally override with client-specific config.
    """
    config = load_yaml(base_path)

    if override_path and os.path.exists(override_path):
        override = load_yaml(override_path)
        config = merge_configs(config, override)

    return config


def merge_configs(base, override):
    """
    Merge two config dictionaries, overriding base keys with override keys.
    """
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            base[k] = merge_configs(base[k], v)
        else:
            base[k] = v
    return base
