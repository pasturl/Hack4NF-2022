import logging

from download_data import download_genie, download_gene_info
from process_data import process_genie_data
from dataset import create_dataset
from ml_model import train_model
from nlp_genes_info import bertopic_to_important_genes
from utils import read_yaml
import time

# Get timestamp
ts = time.time()

logging.basicConfig(format="%(asctime)s - %(levelname)s : %(message)s",
                    datefmt="%y/%m/%d %I:%M:%S %p",
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(f"logs/LOG_{ts}.txt", "a"),
                    ]
                    )
logging.root.setLevel(logging.INFO)
log = logging.getLogger('Hack4NF')


log.info('Reading config files')
# Read configuration files
config_path = "config/resources.yaml"
config = read_yaml(config_path)
genie = config["genie"]
genie["models_path"] = genie["models_path"]+genie["aggregation_level"]+f"/models_{ts}/"
genie["models_metrics_file"] = genie["models_path"]+f"models_{ts}.csv"

# Read synapses credentials
secrets_path = ".secrets/synapses_credentials.yaml"
credentials = read_yaml(secrets_path)

if __name__ == "__main__":
    log.info('Downloading Genie data')
    download_genie(genie, credentials)

    log.info('Processing data')
    process_genie_data(genie)

    log.info('Downloading gene info')
    download_gene_info(genie)

    log.info('Creating dataset')
    datasets = create_dataset(genie)

    if 'binary_classification' in genie["training_mode"]:
        log.info('Training ML supervised model binary classification')
        # Execute all cancer types if targets contains "All"
        if "All" in genie["targets"]:
            with open('data/genie_cancer_types_features.txt') as f:
                cancer_types_cols = f.read().splitlines()
            for target in cancer_types_cols:
                model_lgb = train_model(genie, datasets, target)
        else:
            for target in genie["targets"]:
                model_lgb = train_model(genie, datasets, target)

    bertopic_to_important_genes(genie)
