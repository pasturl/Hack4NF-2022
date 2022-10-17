import synapseclient
import synapseutils
import logging
import os

from utils import create_folder

log = logging.getLogger(__name__)


def download_genie(genie, credentials):
    if os.path.isdir(genie["data_path"]):
        log.info('Genie folder already created, skipping download step')
    else:
        log.info('Creating folders to download Genie data')
        # Create folders to download data
        case_list_path = genie["data_path"]+genie["case_list_folder"]
        gene_panels_path = genie["data_path"]+genie["gene_panels_folder"]

        create_folder(genie["data_path"])
        create_folder(case_list_path)
        create_folder(gene_panels_path)

        log.info('Connecting to Synapses to download data')
        syn = synapseclient.Synapse()
        ## log in using username and password
        syn.login(email=credentials["username"], password=credentials["password"])

        log.info(f'Downloading Genie data Synapses id {genie["synapses_id"]}')
        files = synapseutils.syncFromSynapse(syn, genie["synapses_id"])
        for file in files:
            log.info(f'Downloading Genie file {file.name}')
            if "cases_" in file.name:
                syn.get(file.id, downloadLocation=case_list_path)
            elif "data_gene_panel_" in file.name:
                syn.get(file.id, downloadLocation=gene_panels_path)
            else:
                syn.get(file.id, downloadLocation=genie["data_path"])

