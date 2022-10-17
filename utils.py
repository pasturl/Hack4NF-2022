from pathlib import Path
import yaml


def create_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_yaml(path):
    with open(path, "r") as stream:
        try:
            file = yaml.safe_load(stream)
            return file
        except yaml.YAMLError as exc:
            print(exc)
