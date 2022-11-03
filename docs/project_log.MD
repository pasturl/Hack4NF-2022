
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
      from different files (clinical, samples, mutations variations, patient cancer types)
3. Create dataset
    * Input ->  transformed data with PK
    * Output -> unique dataframe with selected features and targets
4. Lightgbm model
    * Input ->  dataset with selected features and targets
    * Output -> trained a model with binary classification for each target, evaluation metrics, model interpretability plots

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

