# Diaster Response Project
Data science project on text classification for quick aid delivery.

This repository contains material related to a personal project 
that aims to deliver a machine learning based model for text classification
to support the coordination of aid efforts during a natural disaster.

Here, we provide python code for processing raw data, hyperparameter tuning and model evaluation.
The developed machine learning model was also incorporated into a 
web application that is deployed on [Heroku](https://www.heroku.com/home).

## Content
The project is organized into three main folders that contain
the following files:

The folder `/data` contains the raw data in form of two separate csv files (`disaster_categories.csv`, `disaster_messages.csv`).
Running `process_data.py` from the command line merges these two files,
cleans the data and saves the result in the SQLite database file `DisasterResponse.db`.

The folder `/models` contains the final model which is based on a random forest classifier. Evoking `train_classifier.py` loads the database, performs 
hyperparameter tuning via random grid search and evaluates the tuned model on the test set. If the run terminates successfully, three output files are written:
1. `classifier.pkl` corresponds to the saved model
1. `cls_report.pkl` contains the classification report, which lists the precision, recall and F1 score for each class as well as the average values
1. `confusion_matrices.npy` contains the confusion matrix for each class

Further details on the model's hyperparameters and its performace on the test set are summarized [here](www.google.com).

The folder `/app` contains material for creating the web application. In it you find `run.py`, which allows for running the application locally.
The subfolder templates contains html files for the frontend. Note that the design of the user interface relies on the Bootstrap framework.


## Dependencies
This project relies on the following python packages:
 - numpy
 - pandas
 - nltk
 - scikit-learn
 - sqlalchemy 
 - json
 - ploty
 - flask
