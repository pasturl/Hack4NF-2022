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
        genie_processed_data[genie_file] = pd.read_csv(f"{genie_processed_path}{filename}",
                                                       sep=";")
    return genie_processed_data


def join_processed_data_by_patient(processed):
    """
    Create dataset at patient level
    """
    log.info('Creating dataset at patient level')
    df_clinical_sample = processed["clinical_sample"][["PATIENT_ID", "SAMPLE_ID"]]
    log.info('Merging clinical_sample and mutations_extended')
    df_samples = pd.merge(df_clinical_sample,
                          processed["mutations_extended"],
                          on="SAMPLE_ID",
                          how="left")
    df_samples = df_samples.drop(columns=["SAMPLE_ID"])
    df_samples = df_samples.groupby("PATIENT_ID").max()
    log.info('Merging clinical_patient')
    df = pd.merge(processed["clinical_patient"], df_samples,
                  on="PATIENT_ID", how="left")
    log.info('Merging patient_cancer_types')
    df = pd.merge(df, processed["patient_cancer_types"],
                  on="PATIENT_ID", how="left")

    return df


def join_processed_data_by_sample(processed):
    """
    Create dataset at sample level
    """
    log.info('Creating dataset at sample level')
    df_clinical_sample = processed["clinical_sample"][["PATIENT_ID", "SAMPLE_ID"]]
    df_clinical_patient = processed["clinical_patient"][["PATIENT_ID", "AGE_CONTACT", "SEX", "PRIMARY_RACE"]]
    log.info('Merging clinical_sample and mutations_extended')
    df_samples = pd.merge(df_clinical_sample,
                          processed["mutations_extended"],
                          on="SAMPLE_ID",
                          how="left")
    # df_samples = df_samples.drop(columns=["SAMPLE_ID"])
    # df_samples = df_samples.groupby("PATIENT_ID").max()
    log.info('Merging clinical_patient ')
    df = pd.merge(df_samples, df_clinical_patient,
                  on="PATIENT_ID", how="left")
    log.info('Merging sample_cancer_types')
    df = pd.merge(df, processed["sample_cancer_types"],
                  on="SAMPLE_ID", how="left")

    return df


def save_dataset(genie, datasets):
    datasets["dataset_by_patient"].to_csv(genie["dataset_path"] + "dataset_patient.csv", sep=";", index=False)
    datasets["dataset_by_sample"].to_csv(genie["dataset_path"] + "dataset_sample.csv", sep=";", index=False)


def create_dataset(genie):
    dataset_by_patient_file = f"{genie['dataset_path']}dataset_patient.csv"
    dataset_by_sample_file = f"{genie['dataset_path']}dataset_sample.csv"
    datasets = {}
    if (os.path.isfile(dataset_by_patient_file)) & (os.path.isfile(dataset_by_sample_file)):
        log.info(f'Reading dataset by patient from: {dataset_by_patient_file}')
        log.info(f'Reading dataset by sample from: {dataset_by_sample_file}')
        datasets["dataset_by_patient"] = pd.read_csv(dataset_by_patient_file, sep=";")
        datasets["dataset_by_sample"] = pd.read_csv(dataset_by_sample_file, sep=";")
    else:
        log.info('Creating datasets by patient and sample')
        genie_processed = load_genie_processed_data(genie)
        datasets["dataset_by_patient"] = join_processed_data_by_patient(genie_processed)
        datasets["dataset_by_sample"] = join_processed_data_by_sample(genie_processed)
        save_dataset(genie, datasets)
    return datasets
