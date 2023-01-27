# italianVAT

Repository of the italianVAT corpus, a novel corpus that consists of 226 Italian second-instance decisions on Value Added Tax (VAT) by the Regional Tax Commissions from various judicial districts.
The corpus is annotated based on the structural parts of the decisions.

## Citation

If you use this repository, dataset, or code, please cite our work as:

*Galli, Federico & Grundler, Giulia & Fidelangeli, Alessia & Galassi, Andrea & Lagioia, Francesca & Palmieri, Elena & Ruggeri, Federico & Sartor, Giovanni & Torroni, Paolo. (2022). Predicting Outcomes of Italian VAT Decisions. 10.3233/FAIA220465.*

```
@inproceedings{galli2022,
author = {Galli, Federico and Grundler, Giulia and Fidelangeli, Alessia and Galassi, Andrea and Lagioia, Francesca and Palmieri, Elena and Ruggeri, Federico and Sartor, Giovanni and Torroni, Paolo},
year = {2022},
month = {12},
title = {Predicting Outcomes of Italian VAT Decisions1},
isbn = {9781643683645},
doi = {10.3233/FAIA220465},
booktitle = {Legal Knowledge and Information Systems},
volume = {362},
pages = {188--193},
series = {Frontiers in Artificial Intelligence and Applications}
}
```


## Repository structure

* the paper folder contains the paper pdf
* the italianVAT_dataset folder contains the tagged documents in xml format
* xmlToJson.py is a python script that converts the dataset into json format
* create_df.py is a python script that generates the dataframe
* outcomeprediction.py defines the functions that perform the classification task
* run_experiments.py calls the outcomeprediction.py functions with the desired parameters


## How to run the experiments

* run xmlToJson.py to convert the xml dataset into the required json format
* run create_df.py to create the dataframe
* open run_experiments.py to choose the task's parameters, embeddings and classifiers, or run it as it is to get the complete set of experiments
