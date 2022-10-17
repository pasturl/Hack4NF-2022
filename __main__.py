import logging

from download_data import download_genie
from process_data import process_genie_data

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
    #create_dataset(genie)
