import pandas as pd
import numpy as np
import logging
import os
import pickle

from utils import create_folder

log = logging.getLogger(__name__)


def load_genie_raw_data(genie):
    genie_path = genie["data_path"]
    genie_data = {}

    for genie_file in genie["files"]:
        if genie_file in genie["processed_files"]:
            filename = genie["files"][genie_file]["file_name"]
            if "skip_rows" in genie["files"][genie_file]:
                skip_rows = genie["files"][genie_file]["skip_rows"]
            else:
                skip_rows = 0

            genie_data[genie_file] = pd.read_csv(f"{genie_path}/{filename}",
                                                 sep="\t", skiprows=skip_rows)
    return genie_data


def clean_year(year):
    special_cases = ["Not Collected", "Not Released", "Unknown",
                     "withheld", "cannotReleaseHIPAA", "Not Applicable", np.NaN]
    if year == ">89":
        return 90
    elif year == "<18":
        return 17
    elif year in special_cases:
        return None
    else:
        return int(year)


def clean_days(days):
    special_cases = ["Unknown"]
    if days == ">32485":
        return 32485
    elif days == "<6570":
        return 6570
    elif days in special_cases:
        return None
    else:
        return int(days)


def add_age(df_clinical_patient):
    df_clinical_patient["BIRTH_YEAR"] = df_clinical_patient["BIRTH_YEAR"].apply(clean_year)
    df_clinical_patient["YEAR_CONTACT"] = df_clinical_patient["YEAR_CONTACT"].apply(clean_year)
    df_clinical_patient["YEAR_DEATH"] = df_clinical_patient["YEAR_DEATH"].apply(clean_year)
    df_clinical_patient["AGE_CONTACT"] = df_clinical_patient["YEAR_CONTACT"] - df_clinical_patient["BIRTH_YEAR"]
    df_clinical_patient["AGE_DEATH"] = df_clinical_patient["YEAR_DEATH"] - df_clinical_patient["BIRTH_YEAR"]
    df_clinical_patient["AGE_DEATH"] = df_clinical_patient["AGE_DEATH"].apply(lambda x: np.NaN if x < 0 else x)
    return df_clinical_patient


def process_clinical_patient(df_clinical_patient, genie):
    '''
        Process clinical_patient and add age columns
    '''
    clinical_patient_file = genie["processed_data"]+genie["processed_files"]["clinical_patient"]["file_name"]
    if os.path.isfile(clinical_patient_file):
        log.info(f'Genie clinical_patient already created, skipping step')
        df_clinical_patient = pd.read_csv(clinical_patient_file,
                                                       sep=";")
    else:
        df_clinical_patient = add_age(df_clinical_patient)
    return df_clinical_patient


def process_clinical_sample(df_clinical_sample, genie):
    '''
    Process clinical_sample and add age columns
    '''
    clinical_sample_file = genie["processed_data"] + genie["processed_files"]["clinical_sample"]["file_name"]
    if os.path.isfile(clinical_sample_file):
        log.info(f'Genie clinical_sample already created, skipping step')
        df_clinical_sample = pd.read_csv(clinical_sample_file,
                                          sep=";")
    else:
        df_clinical_sample["AGE_AT_SEQ_REPORT_DAYS"] = df_clinical_sample["AGE_AT_SEQ_REPORT_DAYS"].apply(clean_days)
        df_clinical_sample["AGE_AT_SEQ_REPORT"] = df_clinical_sample["AGE_AT_SEQ_REPORT_DAYS"].apply(lambda x: x//365)
    return df_clinical_sample


def process_patient_cancer_types(df_clinical_sample, genie):
    '''
    Process clinical_sample and generate one hot encoding with each patient_cancer_types as column
    '''
    patient_cancer_types_file = genie["processed_data"] + genie["processed_files"]["patient_cancer_types"]["file_name"]
    if os.path.isfile(patient_cancer_types_file):
        log.info(f'Genie patient_cancer_types already created, skipping step')
        df_cs = pd.read_csv(patient_cancer_types_file,
                                         sep=";")
    else:
        df_cs = df_clinical_sample[["PATIENT_ID", "CANCER_TYPE"]].copy()
        df_cs = pd.get_dummies(df_cs.set_index('PATIENT_ID'), prefix='', prefix_sep='')
        df_cs.reset_index(inplace=True)
        df_cs = df_cs.groupby("PATIENT_ID").max()
        df_cs.reset_index(inplace=True)
        save_cancer_types_columns(df_cs)
    return df_cs


def group_by_chunk(df_mut, col_group, n_rows):
    if len(df_mut) % n_rows != 0:
        chunks = len(df_mut)//n_rows+1
    else:
        chunks = len(df_mut) // n_rows
    df = pd.DataFrame()
    for chunk in range(chunks):
        log.info(f'Chunk {chunk+1}/{chunks}')
        df_tmp = df_mut[chunk*n_rows:chunk*n_rows+n_rows]
        df_tmp = df_tmp.groupby(col_group).max()
        df = pd.concat([df, df_tmp])
    df = df.groupby(col_group, observed=True).max()
    df.reset_index(inplace=True)
    return df


def process_mutations_variants(df_mutation, genie):
    '''
    Process mutations_extended and generate one hot encoding with each Hugo Symbols and variant type as column
    '''
    mutations_extended_file = genie["processed_data"] + genie["processed_files"]["mutations_extended"]["file_name"]
    if os.path.isfile(mutations_extended_file):
        log.info(f'Genie mutations_extended already created, skipping step')
        df_mut_process = pd.read_csv(mutations_extended_file,
                            sep=";")
    else:
        df_mut_var = df_mutation[["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Type"]].copy()
        df_mut_var["Hugo_Symbol_variant"] = df_mut_var["Hugo_Symbol"] + "_" + df_mut_var["Variant_Type"]
        df_mut_var = df_mut_var[["Hugo_Symbol_variant", "Tumor_Sample_Barcode"]]
        df_mut_process = pd.get_dummies(df_mut_var.set_index('Tumor_Sample_Barcode'), prefix='', prefix_sep='')
        df_mut_process.reset_index(inplace=True)
        df_mut_process = df_mut_process.rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})
        df_mut_process = df_mut_process.sort_values("SAMPLE_ID")
        # This group by generate a memory error
        # So the groupby is done by chunk
        # df_mut_process = df_mut_process.groupby("SAMPLE_ID", observed=True).max()
        col_group = "SAMPLE_ID"
        n_rows = 100000
        df_mut_process = group_by_chunk(df_mut_process, col_group, n_rows)
        # df_dd = dd.from_pandas(df_mut_process, npartitions=12)
        # result = df_dd.groupby('SAMPLE_ID').max().reset_index().compute()
        # df_mut_process.reset_index(inplace=True)
        save_mutations_columns(df_mut_process)

    return df_mut_process


def save_mutations_columns(df):
    mutation_cols = list(df.columns)
    mutation_cols.pop(0)
    # TODO parametrize path
    with open('data/genie_mutations_features.txt', 'w') as f:
        for mutation in mutation_cols:
            f.write(f"{mutation}\n")


def save_cancer_types_columns(df):
    cancer_types_cols = list(df.columns)
    cancer_types_cols.pop(0)
    # TODO parametrize path
    with open('data/genie_cancer_types_features.txt', 'w') as f:
        for cancer_types in cancer_types_cols:
            f.write(f"{cancer_types}\n")


def process_mutations(df_mutation, genie):
    '''
    Process mutations_extended and generate one hot encoding with each Hugo Symbols as column
    '''
    mutations_extended_file = genie["processed_data"] + genie["processed_files"]["mutations_extended"]["file_name"]
    if os.path.isfile(mutations_extended_file):
        log.info(f'Genie mutations_extended already created, skipping step')
        df_mut_process = pd.read_csv(mutations_extended_file,
                                     sep=";")
    else:
        df_mut = df_mutation[["Hugo_Symbol", "Tumor_Sample_Barcode"]].copy()
        df_mut_process = pd.get_dummies(df_mut.set_index('Tumor_Sample_Barcode'), prefix='', prefix_sep='')
        df_mut_process.reset_index(inplace=True)
        df_mut_process = df_mut_process.rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})
        df_mut_process = df_mut_process.groupby("SAMPLE_ID").max()
        df_mut_process.reset_index(inplace=True)
        save_mutations_columns(df_mut_process)

    return df_mut_process


def save_process_data(genie, genie_process):
    for data_process in genie_process:
        log.info(f'Saving Genie {data_process}')
        genie_process[data_process].to_csv(f"{genie['processed_data']}{data_process}.csv", sep=";", index=False)


def process_genie_data(genie):
    # TODO Check if all files exist to avoid read and write
    # if os.path.isdir(genie["processed_data"]):
    #     log.info('Genie processed folder already created, skipping processing step')
    create_folder(genie["processed_data"])
    log.info('Loading Genie raw data')
    genie_process = {}
    genie_data = load_genie_raw_data(genie)

    log.info('Processing clinical patient data')
    genie_process["clinical_patient"] = process_clinical_patient(genie_data["clinical_patient"], genie)

    log.info('Processing clinical sample data')
    genie_process["clinical_sample"] = process_clinical_sample(genie_data["clinical_sample"], genie)

    log.info('Processing clinical sample data to create patient_cancer_types')
    genie_process["patient_cancer_types"] = process_patient_cancer_types(genie_data["clinical_sample"], genie)

    log.info('Processing mutations data')
    genie_process["mutations_extended"] = process_mutations_variants(genie_data["mutations_extended"], genie)
    # genie_process["mutations_extended"] = process_mutations(genie_data["mutations_extended"])

    log.info('Saving Genie processed data')
    save_process_data(genie, genie_process)
