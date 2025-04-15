import yaml

def load_yaml(file_path):
    with open(file_path) as f:
        cfg = yaml.safe_load(f)

    return cfg


class Config:
    yaml_file_path = ""

    cfg = {}

    @classmethod
    def update_attrs(cls, **kwargs):
        for key, value in kwargs.items():
            setattr(cls, key, value)

        if cls.yaml_file_path:
            print(f"Loading YAML file from: {cls.yaml_file_path}")
            cls.cfg = load_yaml(cls.yaml_file_path)

Config.update_attrs(yaml_file_path="./config/params.yaml")
        