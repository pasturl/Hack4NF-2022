import logging

from download_data import download_genie
from process_data import process_genie_data
from dataset import create_dataset
from ml_model import train_model

from utils import read_yaml

logging.basicConfig()
logging.root.setLevel(logging.INFO)
log = logging.getLogger('Hack4NF')


log.info('Reading config files')
# Read configuration files
config_path = "config/resources.yaml"
config = read_yaml(config_path)
genie = config["genie"]


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
        for target in genie["targets"]:
            model_lgb = train_model(genie, dataset, target)
