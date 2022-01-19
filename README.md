<p align="center">
  <a><img alt="Python 3.8 badge" src="https://img.shields.io/badge/Pyhon-3.8-blue?&logo=python&logoColor=yellow"></a>
  <a href="https://github.com/chris-hoffmann/dog-breed-web-app/blob/master/LICENSE"><img alt="MIT License badge" src="https://img.shields.io/badge/License-MIT-blue"> </a>
  <br>
  <a><img alt="SQLite" src="https://img.shields.io/badge/SQLite-003B57?logo=SQLite&logoColor=whit"> </a>
  <a><img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white"> </a>
  <a><img alt="Plotly" src="https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white"> </a>
  <a><img alt="Scikit" src="https://img.shields.io/badge/Scikit-F7931E?logo=scikit-learn&logoColor=white"> </a>
  <br>
  <a><img alt="Bootstrap" src="https://img.shields.io/badge/Bootstrap-7952B3?logo=bootstrap&logoColor=white"> </a>
  <a><img alt="Flask" src="https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white&style=flat"> </a>
  <a><img alt="Heroku" src="https://img.shields.io/badge/Heroku-430098?logo=heroku&logoColor=white"> </a>
</p>

# Disaster Response Project
Applied machine learning project on text classification for quick aid delivery.

This repository contains material related to a previous class project that aimed to deliver a machine learning-based model for text classification to support the coordination of aid efforts during a natural disaster.

The developed model was also incorporated into a web application that is available at
<p align="center">
  <a href="https://disaster-response-txt-app.herokuapp.com"><strong>https://disaster-response-txt-app.herokuapp.com</strong></a>
</p>

This branch contains further information regarding the modeling including code for data processing, hyperparameter tuning, and model evaluation.

## Requirements
To install requirements into a conda environment:

```setup
conda env create --file environment.yml
conda activate txt_app
pip install -r requirements.txt
pip install -e . 
```

## Data processing 
The underlying data are publicly available from [here](https://appen.com/datasets/combined-disaster-response-data) but also included in this repository (see [`data/`](data)).

To prepare the raw data for modeling:

```process
python scripts/process_data.py --inputs data/disaster_messages.csv --labels data/disaster_categories.csv --output data/disaster_response.db
```

The script outputs a SQLite database that contains the merged and cleaned dataset. 

## Training
To train a text classifier including hyperparameter tuning via random search:
```train rf
python scripts/train_eval.py --database data/disaster_response.db 
```

If the run terminates successfully, three output files are written to a subfolder that
resides in [`experiments/`](experiments/):
1. `<classifier>_model.pkl` corresponds to the saved model
1. `cls_report.pkl` contains the classification report, which lists the precision, recall, and F1 score for each class as well as the average values
1. `confusion_matrices.npy` contains the confusion matrices for each class

The default training scheme relies on a logistic regression (logit) classifier. To train a Random Forest classifier, use the `--clf` flag:

```train logit
python scripts/train_eval.py --database data/disaster_response.db --clf random_forest
```

## Evaluation 
We compare the classification performance of both models (random forests vs. logit) on a hold-out set in an accompanying notebook (see [`notebooks/model_comparison.ipynb`](/notebooks)).

## Pre-trained models
The pre-trained logit classifier can be found in the [`master branch`](https://github.com/chris-hoffmann/disaster_response/tree/master/web_app/static/model).
The pre-trained random forest classifier (~560 MB) is available via [`GitHub Releases`](https://github.com/chris-hoffmann/disaster_response/releases).

## Modeling details
Feature extraction from text messages followed standard protocols: Word tokens were transformed to their base form using the WordNet Lemmatizer. Then, a count matrix of word tokens was generated. This matrix was subsequently converted into a tf-idf (term frequency times inverse document frequencies) representation that was employed for modeling. Several classifiers were tested using Scikit-learn. The parameters for feature extraction and classification were further refined by applying a random hyperparameter search. For the deployed model, a classifier based on logistic regression was chosen. More information regarding the selected hyperparameters and the performance of models determined on a hold-out set can be found in a dedicated notebook (`model_comparison.ipynb`).

## Repository structure
```
.
├── environment.yml
├── requirements.txt
├── setup.py
├── data
│   ├── disaster_categories.csv
│   └── disaster_messages.csv
├── experiments
│   ├── logit_2021-12-19_19-23-37
│   │   ├── cls_report.pkl 
│   │   └── confusion_matrices.npy
│   └── random_forest_2021-12-19_20-51-13
│       ├── cls_report.pkl
│       └── confusion_matrices.npy
├── notebooks
│   └── model_comparison.ipynb
├── scripts
│   ├── process_data.py
│   └── train_eval.py
└── web_app
    ├── __init__.py
    ├── static
    └── utils.py
```