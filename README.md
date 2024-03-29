<p align="center">
  <a href="https://heroku.com">
    <img src="web_app_snapshot.PNG" alt="Snapshot of web app">
  </a>
</p>

# Disaster Response Project
Data science project on text classification for quick aid delivery.

This repository contains material related to a personal project 
that aims to deliver a machine learning-based model for text classification
to support the coordination of aid efforts during a natural disaster.

Here, we provide python code for processing raw data, hyperparameter tuning, and model evaluation.
The developed machine learning model was also incorporated into a 
web application that is deployed on [Heroku](https://www.heroku.com/home).

## Content
The project is organized into three main folders (`data`, `code`, `models`) that contain
the following files:

```
|-- data
|   |-- disaster_categories.csv
|   |-- disaster_messages.csv
|   `-- disaster_response.db
|-- code
|   |-- __init__.py
|   |-- app
|   |   |-- __init__.py
|   |   |-- templates
|   |   |   |-- about_the_model.html
|   |   |   |-- go.html
|   |   |   `-- master.html
|   |   `-- utils.py
|   |-- process_data.py
|   |-- train_classifier.py
|   |-- run_app.py
|   `-- model_details.ipynb
`-- models
    |-- classifier.pkl
    |-- cls_report.pkl
    `-- confusion_matrix.npy
```

The folder `/data` contains the raw data in the form of two separate csv files (`disaster_categories.csv`, `disaster_messages.csv`). Running ```python /code/process_data.py``` from the command line merges these two files, cleans the data, and saves the result in the SQLite database file `disaster_response.db`.

The folder `/models` contains the final model, which is based on a random forest classifier. Running ```python /code/train_classifier.py``` loads the database, performs hyperparameter tuning via random grid search, and evaluates the tuned model on the test set. If the run terminates successfully, three output files are written:
1. `classifier.pkl` corresponds to the saved model
1. `cls_report.pkl` contains the classification report, which lists the precision, recall, and F1 score for each class as well as the average values
1. `confusion_matrix.npy` contains the confusion matrices for each class

Further details on the model's hyperparameters and its performace on the test set are summarized in the notebook [`/code/model_details.ipynb`](https://github.com/chris-hoffmann/disaster_response/blob/master/code/model_details.ipynb).

The folder `/code/app` contains material for creating the web application. In it, you find the subfolder `/templates` that contains the html files for the frontend. Note that the design of the user interface takes advantage of the Bootstrap framework. You can also run the app locally by invoking ```python/code/run_app.py```. 

## Dependencies
This project relies on the following python packages:
 - sqlalchemy 
 - numpy
 - pandas
 - nltk
 - scikit-learn
 - ploty
 - flask
