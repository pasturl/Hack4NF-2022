import pandas as pd
import numpy as np
import logging
import os

from utils import create_folder

log = logging.getLogger(__name__)


def load_genie_processed_data(genie):
    genie_processed_path = genie["processed_data"]
    genie_processed_data = {}
    for genie_file in genie["processed_files"]:
        filename = genie["processed_files"][genie_file]["file_name"]

        genie_processed_data[genie_file] = pd.read_csv(f"{genie_processed_path}/{filename}",
                                                       sep=";")
    return genie_processed_data


def join_processed_data(processed):


def create_dataset(genie):
    genie_processed = {}
    genie_processed = load_genie_processed_data(genie)
    df = join_processed_data(genie_processed)
