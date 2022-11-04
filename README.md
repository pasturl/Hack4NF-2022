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


# Problem definiton
Two approach are proposed:
1. Use genetic mutations to predict the cancer (binary classification and multiclass).
2. Use genetic mutations to predict other mutations without using the target
   as feature. It could help to understand gene mutation correlation 
   with NF1, NF2, SMARCB1 and LZTR1.This trained model could be used as base 
   to predict cancer types using transfer learning.

# Next steps
* Refine which features are used 
* Analyze which genes are evaluated in each study
* Evaluate how are handle multi samples from a patient
* Implement a multiclass model
* Develop an embedding of mutations to be used as pretrained model
* Improve text cleaning adding more stopwords
* Review most important genes from each cancer type
* Understand biological function of important genes
* Evaluate possible genetic drug targets

# References
* https://www.mskcc.org/cancer-care/types/neurofibromatosis/neurofibromatosis-type-1-nf1
* https://www.mayoclinic.org/es-es/diseases-conditions/neurofibromatosis/symptoms-causes/syc-20350490
* https://www.nhs.uk/conditions/neurofibromatosis-type-1/
* https://www.thieme-connect.com/products/ejournals/abstract/10.1055/s-0034-1382021
* https://pubmed.ncbi.nlm.nih.gov/25062113/
