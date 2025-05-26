import yaml, pathlib
def load_config(path):
    with open(pathlib.Path(path), "r") as f:
        return yaml.safe_load(f)
