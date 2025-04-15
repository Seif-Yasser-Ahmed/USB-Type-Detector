import yaml

def load_yaml(file_path):
    with open(file_path) as f:
        cfg = yaml.safe_load(f)

    return cfg


class Config:
    yaml_file_path = "../../config/params.yaml"
    cfg = load_yaml(yaml_file_path)