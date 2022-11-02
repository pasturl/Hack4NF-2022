import synapseclient
import synapseutils
import logging
import os
import pandas as pd
import httplib2 as http
import json
import time

try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

from utils import create_folder

log = logging.getLogger(__name__)


def download_genie(genie, credentials):
    log.info('Downloading Genie data')
    if os.path.isdir(genie["data_path"]):
        log.info('Genie folder already created, skipping download step')
    else:
        log.info('Creating folders to download Genie data')
        # Create folders to download data
        case_list_path = genie["data_path"] + genie["case_list_folder"]
        gene_panels_path = genie["data_path"] + genie["gene_panels_folder"]

        create_folder(genie["data_path"])
        create_folder(case_list_path)
        create_folder(gene_panels_path)

        log.info('Connecting to Synapses to download data')
        syn = synapseclient.Synapse()
        # log in using username and password
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


def download_gene_info(genie):
    if os.path.isfile(genie["processed_data"] + genie["processed_files"]["gene_info"]["file_name"]):
        log.info('Gene info already download, skipping step')
    else:
        df_genes = pd.read_csv(genie["data_path"] + "data_mutations_extended_13.3-consortium.txt",
                               sep="\t")
        df_genes = df_genes["Hugo_Symbol"].drop_duplicates().reset_index(drop=True)
        df_genes = pd.DataFrame(df_genes)
        # df_genes["gene_info"] = df_genes["Hugo_Symbol"].apply(lambda x: find_gene_symbol(x))
        genes = list(df_genes["Hugo_Symbol"].values)
        total_genes = len(genes)
        info_genes = []
        n_gene = 1
        for gene in genes:
            log.info(f'Downloading gene {gene}: {n_gene}/{total_genes}')
            data_gene = find_gene_symbol(gene)
            info_genes.append(data_gene)
            time.sleep(0.01)
            n_gene+=1
        df_genes["gene_info"] = info_genes
        df_genes["gene_info"] = df_genes["gene_info"].fillna("NA")
        df_genes["title"] = df_genes["gene_info"].apply(
            lambda x: x["title"] if 'title' in x else None)
        df_genes["status"] = df_genes["gene_info"].apply(
            lambda x: x["status"] if 'status' in x else None)
        df_genes["description"] = df_genes["gene_info"].apply(
            lambda x: x["description"] if 'description' in x else None)
        cols_to_save = ["Hugo_Symbol", "title", "status", "description"]
        df_genes[cols_to_save].to_csv(genie["processed_data"] + genie["processed_files"]["gene_info"]["file_name"],
                                      sep=";", index=False)


def find_gene_symbol(gene):
    uri = 'http://marrvel.org/data'
    path = f'/omim/gene/symbol/{gene}'

    target = urlparse(uri + path)
    method = 'GET'
    body = ''
    headers = {
        'Accept': 'application/json',
    }

    h = http.Http()

    response, content = h.request(
        target.geturl(),
        method,
        body,
        headers)
    data = json.loads(content)
    return data
