import yaml
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    current_file = Path(__file__)
    # Go up 3 levels from src/utils/config.py to reach project root
    project_root = current_file.parent.parent.parent
    return project_root


def load_config(config_name='parameters.yaml'):
    """Load configuration from YAML file"""
    project_root = get_project_root()
    config_path = project_root / 'config' / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def save_config(config, config_name='parameters.yaml'):
    """Save configuration to YAML file"""
    project_root = get_project_root()
    config_path = project_root / 'config' / config_name

    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Configuration saved to {config_path}")


def get_data_paths():
    """Get standardized data paths"""
    config = load_config()
    project_root = get_project_root()

    paths = {
        'raw_data': project_root / config['data']['raw_data_path'],
        'processed_data': project_root / config['data']['processed_path'],
        'models': project_root / 'models'
    }

    # Create directories if they don't exist
    for path in paths.values():
        if path.is_dir():
            path.mkdir(parents=True, exist_ok=True)

    return paths


def update_config(key, value, config_name='parameters.yaml'):
    """Update a specific configuration value"""
    config = load_config(config_name)

    # Navigate nested keys (e.g., 'model.hyperparameters.n_estimators')
    keys = key.split('.')
    current_level = config

    for k in keys[:-1]:
        if k not in current_level:
            current_level[k] = {}
        current_level = current_level[k]

    current_level[keys[-1]] = value

    save_config(config, config_name)
    print(f"Updated config: {key} = {value}")


if __name__ == "__main__":
    # Test the config loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Project root: {get_project_root()}")

    data_paths = get_data_paths()
    print("Data paths:")
    for name, path in data_paths.items():
        print(f"  {name}: {path}")