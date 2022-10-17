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
    '''
    TODO Evaluate how to work with the different sample per patient
    '''
    df_clinical_sample = processed["clinical_sample"][["PATIENT_ID", "SAMPLE_ID"]]
    df_samples = pd.merge(df_clinical_sample,
                          processed["mutations_extended"],
                          on="SAMPLE_ID",
                          how="left")
    df_samples = df_samples.drop(columns=["SAMPLE_ID"])
    df_samples = df_samples.groupby("PATIENT_ID").max()
    df = pd.merge(processed["clinical_patient"], df_samples,
                  on="PATIENT_ID", how="left")
    return df


def save_dataset(genie, df):
    df.to_csv(genie["dataset_path"]+"dataset.csv", sep=";")


def create_dataset(genie):
    genie_processed = {}
    genie_processed = load_genie_processed_data(genie)
    dataset = join_processed_data(genie_processed)
    save_dataset(genie, dataset)
    return dataset
