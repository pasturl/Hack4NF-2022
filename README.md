# Hack4NF-2022
Hackathon for Neurofibromatosis

https://hack4nf-platform.bemyapp.com/#/event 

![hack4NF](static/image/hack4NF.png)

Challenge #1

GENIE-NF tumor identification and classification challenge

Question:

Use the provided Genomics Evidence Neoplasia Information Exchange (GENIE) datasets provided to develop a new framework that accurately uses genomic data to classify tumor samples for neurofibromatosis-related tumors. First, identify the neurofibromatosis-related tumors in the dataset - this could be defined as all tumors with mutations in NF-relevant genes, or tumors that are common in patients with NF, or another rational definition. Then, use one or more classification methods to classify the tumor samples into different groups based on genetic features. 

Dataset(s):

Early access to version 13 of the GENIE dataset for hackathon registrants. You are encouraged to also use other external datasets, but you must include the GENIE dataset in your project.  

Example Solutions:

A classification algorithm that differentiates different types of NF1, NF2, and schwannomatosis-related tumors using clinical sequencing data. A list of the most important features in your algorithm for differentiating tumor types.  

# Questions
* [DATA] What genes are evaluated in each study?
* [MEDICAL] Which types of cancer are related with neurofibromatosis?

# Team methodology
* IMPORTANT Document everything! 
* Be agile and get an MVP as soon as possible!
* Work in one git (https://github.com/pasturl/Hack4NF-2022)
  repository with the code and all documentation
* Each person will work in one branch with an explicable name 
  (i.e. model_multiclass, process_cna, external_data, document features)
* Doing the task commits will be made to explain the changes in 
  the code or the new documentation added. 
* It's recommended made at least one commit at the end of the day
  if you were working in something. 
* After finnish the task, the person will do a Pull Request (PR), 
  and the teammates will review the work to understand it
  and give feedback.

# Problem definiton
Two approach are proposed:
1. Use genetic mutations to predict the cancer (binary classification and multiclass).
2. Use genetic mutations to predict other mutations without using the target
   as feature. It could help to understand gene mutation correlation 
   with NF1, NF2, SMARCB1 and LZTR1.This trained model could be used as base 
   to predict cancer types using transfer learning.

# Project Log

## TODO 2022-10-18
* Create database to store data. The current dataset has 2.8Gb and
  it isn't efficient have it loaded in memory'
* Define data processing to generate features. 
  * Currently, working with gene mutations (using binary feature 
    of mutation and one hot encoding of types of variants
  * It is necessary to define
* Define target based in tumor related with NF:
  * Right now target is the NF1 and NF2 mutations features.
    Binary classification for each NF1/NF2 mutations feature
    with independent models.
  * The challenge is to use a multi-classification (one model to 
    predict different targets and multi-label (one patient/sample 
    can have more than one mutation)))  
  * NF1: cases_peripheral_nervous_system, 
* Create patient and sample dataset 
* Define which model and ML methodology to use.
  * The model used in the first version is Lightgbm, and the
    problem approach as binary classification of each mutation independently.
  * More complex model could be used, i.e. XGBoost multiclass and
    multi-label https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html
  * There are other approach that could be evaluated:
    * OneVsRest https://gabrielziegler3.medium.com/multiclass-multilabel-classification-with-xgboost-66195e4d9f2d
    * Embedding
* Evaluate how to work with the different sample per patient. 
  Current approach use groupby and max of mutations in all sample of a patient
* External data - Other synapses dataset, APIs, webs
  (https://depmap.org/portal/download/), csv
* Add log file with training report of each model
* Organize and prepare notebooks to show reports

## DONE 2022-10-18
* Use mutations variants
* The Genie dataset has over 130k patients, so to process the mutation
  file it is necessary a lot of RAM memory. At least 16GB and 64GB of 
  virtual memory of SSD
  https://www.windowscentral.com/how-change-virtual-memory-size-windows-10.
* Evaluate multiclass with multilabel (predict all mutations with one model)
### Create package with pipeline
1. Download data: 
    * Input -> 
        * Internal data - synapse id
    * Output -> structured csv files
2. Process data: 
    * Input -> structured csv files
    * Output -> transformed data with PK id and all features
      from different files (clinical, samples, mutations variations)
3. Create dataset
    * Input ->  transformed data with PK
    * Output -> unique dataframe with selected features and filtered data
4. Lightgbm model
    * Input ->  unique dataframe with selected features and filtered data
    * Output -> trained model, evaluation metrics, model interpretability plots

## DONE 2022-10-14
* Exploratory Data Analyses with genie v12:
  * Evaluate types of cancer related with mutations in NF1, NF2, SMARCB1 and/or LZTR1
* Process mutation and CNA data
* Non supervised techniques to cluster patients
* Analyze important features by cluster 
* Subsample dataset with relevant 
* Supervised ML Model (LightGBM) to predict mutations (NF1, NF2, SMARCB1 and LZTR1)
* Define model target
* Check shap feature importance
* BERT:
  * Input -> the list of important features and gen info from
    'http://marrvel.org/data/omim/gene/symbol/{gene}'.
  * Output -> topics, key words by topic, tree of topics.


# References
* https://www.mayoclinic.org/es-es/diseases-conditions/neurofibromatosis/symptoms-causes/syc-20350490
* https://www.nhs.uk/conditions/neurofibromatosis-type-1/
* https://www.thieme-connect.com/products/ejournals/abstract/10.1055/s-0034-1382021
* https://pubmed.ncbi.nlm.nih.gov/25062113/