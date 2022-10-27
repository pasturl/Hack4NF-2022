import logging

from download_data import download_genie
from process_data import process_genie_data
from dataset import create_dataset
from ml_model import train_model
from utils import read_yaml
import time

# Get timestamp
ts = time.time()

logging.basicConfig(filename=f"logs/LOG_{ts}.txt")
logging.root.setLevel(logging.INFO)
log = logging.getLogger('Hack4NF')


log.info('Reading config files')
# Read configuration files
config_path = "config/resources.yaml"
config = read_yaml(config_path)
genie = config["genie"]
genie["models_path"] = genie["models_path"]+f"models_{ts}/"
genie["models_metrics_file"] = genie["models_path"]+f"models_{ts}.csv"

# Read synapses credentials
secrets_path = ".secrets/synapses_credentials.yaml"
credentials = read_yaml(secrets_path)

if __name__ == "__main__":
    log.info('Downloading data')
    download_genie(genie, credentials)

    log.info('Processing data')
    process_genie_data(genie)

    log.info('Creating dataset')
    dataset = create_dataset(genie)

    log.info('Filter out Na in target columns')
    # TODO Analyze mutations informed by each study and clean in origin
    dataset = dataset.dropna(subset=genie["targets"])

    if 'binary_classification' in genie["training_mode"]:
        log.info('Trainning ML supervised model binary classification')
        # Execute all cancer types if targets contains "All"
        if "All" in genie["targets"]:
            with open('data/genie_cancer_types_features.txt') as f:
                cancer_types_cols = f.read().splitlines()
            for target in cancer_types_cols:
                model_lgb = train_model(genie, dataset, target)
        else:
            for target in genie["targets"]:
                model_lgb = train_model(genie, dataset, target)
