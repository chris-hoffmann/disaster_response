# Diaster Response Project
Data science project on text classification for quick aid delivery.

This repository contains material related to a personal data science project 
that aims to deliver a machine learning based model for text classification
to support the coordination of aid efforts during a natural disaster.

Here, we provide python code for processing raw data, hyperparameter tuning and model evaluation.
The developed machine learning model was also incorporated into a 
web application that is deployed on Heroku.

## Content
The project is organized into three main folders that contain
the following files:

The folder `/data` contains the raw data in form of two separate csv files.
Running the `process_data.py` from the command line merges these two files,
cleans the data and saves the result in the SQLite database file .

The folder `/models` contains the final model which is based on a random forest classifier. Evoking `train_classifier.py` loads the database, performs 
hyperparameter screening and evaluates the tuned model on the test set. If the run terminates successfully, two output files are written:
corresponds to the classification report, which lists the precision, recall and F1 Score for each class. The second output file contains
the confusion matrix 

More details on the model's hyperparameters
and it's performace on the test set can be found here.
The script saves the model as pickle file and also outputs the confusion matrix for each class saved as a NumPy binary file and the classification report
as pickle object. The latter lists the precision, recall and F1 score of each class as well as the averaged measures.

The folder `/app` contains material for creating the web application. 
In it you find `run.py`, which allows for running the application locally.
The subfolder templates contains html files for the frontend. The design of
the user interface relies on the Bootstrap framework.


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
