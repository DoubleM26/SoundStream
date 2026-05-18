import yaml

config = None


def load_config(config_file):
    global config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        return config
