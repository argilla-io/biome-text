import yaml


def yaml_to_dict(filepath: str):
    with open(filepath) as yaml_content:
        config = yaml.load(yaml_content)
    return config