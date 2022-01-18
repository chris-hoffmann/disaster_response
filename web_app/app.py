"""Flask application for machine learning-based classification of text messages."""

import os
import json
import plotly
import pandas as pd
import numpy as np
from typing import Sequence
from flask import Flask, render_template, request, jsonify
import joblib
import nltk
from sqlalchemy import create_engine
from web_app.utils import (
    tokenize,
    filter_labels,
    plot_counts_per_label,
    plot_freq_of_messages_per_number_of_labels,
    plot_co_occurence_heatmap,
    get_cooc_matrix,
)


# nltk.download("punkt")
# nltk.download('wordnet')

# fapp = Flask(__name__, template_folder=os.path.abspath("./app/templates"))
# app = Flask(__name__, template_folder=os.path.abspath("templates"))
app = Flask(__name__)


# sqlalchemy.engine.base.Engine

# load model
model = joblib.load("web_app/static/model/logit_clf.pkl")

# load data for plotting
# engine = create_engine("sqlite:///../data/disaster_response.db")
# engine = create_engine("sqlite:///./static/data/disaster_response.db")
engine = create_engine("sqlite:///./web_app/static/data/disaster_response.db")
df: pd.DataFrame = pd.read_sql_table("disaster_response", engine)
df = filter_labels(df)
df_labels: pd.DataFrame = df.drop(df.columns[:4], axis=1)

# generate variables for Figure 1 'Number of messages per label'
label_names: Sequence[str] = [
    label.replace("_", " ") for label in df_labels.columns.values
]
counts_per_label: np.ndarray = df_labels.sum().values

# generate variables for Figure 2 'Histogram of the number of labels per \
# message'

hist_labels_per_msg: pd.Series = df_labels.sum(axis=1).value_counts()

cooc_matrix = get_cooc_matrix(df_labels)

# generate variables for Figure 3 'Co-occurrence heatmap'


@app.route("/")
@app.route("/index")
def index():
    """Generates an index html page that displays plots summarizing the dataset
    and accepts user input for classifying text messages.
    """

    # generate a list of plotly graph objects
    graphs = plot_counts_per_label(counts_per_label, label_names)
    graphs.append(
        plot_freq_of_messages_per_number_of_labels(
            hist_labels_per_msg.index, hist_labels_per_msg.values
        )
    )
    graphs.append(plot_co_occurence_heatmap(label_names, label_names, cooc_matrix))

    # encode plotly graph objects in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


@app.route("/go")
def go():
    """Generates a html page that handles user queries and displays prediction
    results.
    """

    # save user input in query
    query = request.args.get("query", "")

    # use model to perform classification for query text
    classification_labels = model.predict([query])[0]

    # remove child_alone label!
    classification_results = dict(zip(df_labels.columns, classification_labels))

    # This will render the go.html - see template for further information.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


@app.route("/about/")
def about():
    """Generates a html page that informs about the project and the underlying
    machine learning model.
    """
    return render_template("about_the_model.html")


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
