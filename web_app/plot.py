"""Contains utility functions and application code for generating graph objects, 
which are then plotted in the web app using plotly"""

from typing import Sequence

import numpy as np
import pandas as pd
from plotly.graph_objs import Bar, Heatmap
from sqlalchemy import create_engine

from web_app.utils import filter_labels

DATABASE_PATH = "web_app/static/data/disaster_response.db"


def get_plot_dataframe(database_file: str) -> pd.DataFrame:
    """Returns a dataframe of shape $N \times K$ with entries in binary format, where $N$ is the number of samples
        and $K$ is the number of classes.

        The output is then used for generating plots that summarize the training data as displayed on index.html

    Args:
        database_file: Dataframe containing the pre-processed text data.

    Returns:
        df_labels: Dataframe of shape $K \times N$ whose columns correspond to class labels and entries are in binary format. Note that this output analogous to applying the MultiLabelBinarizer from sklearn.
    """

    sql_path = "sqlite:///" + DATABASE_PATH
    engine = create_engine(sql_path)
    table_name = DATABASE_PATH.replace("/", ".").split(".")[-2]
    df: pd.DataFrame = pd.read_sql_table(table_name, engine)
    df = filter_labels(df)
    df_labels: pd.DataFrame = df.drop(df.columns[:4], axis=1)
    return df_labels


def get_cooc_matrix(df_labels: pd.DataFrame) -> np.ndarray:
    """Returns the symmetric pair-wise co-occurrence matrix.

    Args:
        df_label: DataFrame with columns corresponding to class labels.

    Returns:
        cooc_matrix: 2D np.array representing the co-occurrence heatmap of class labels. The shape is $k \times k$, where $k$ refers to the number of classes.
    """

    k = df_labels.shape[-1]
    co_occ_matrix = np.zeros((k, k))
    for i, la in enumerate(df_labels.columns.values):
        df_temp = df_labels.loc[df_labels[la] == 1]
        n, _ = df_temp.shape
        co_occ_matrix[i, :] = df_temp.sum(axis=0).values / n
    return co_occ_matrix


def plot_counts_per_label(x, y):
    """
    Returns a list that contains a dictionary to specify a plotly graph
    object for plotting the number of messages (y) per label (x) as a
    horizontal bar plot using Plotly
    """

    graph_dict = {
        "data": [Bar(x=x, y=y, orientation="h")],
        "layout": {
            "title": "Number of messages per label",
            "autosize": False,
            "height": 700,
            "width": 500,
            "yaxis": {
                "automargin": True,
                "title": "Label",
                "dtick": 1,
            },
            "xaxis": {"title": "Number of messages"},
        },
    }
    return graph_dict


def plot_freq_of_messages_per_number_of_labels(x, y):
    """
    Returns a dictionary that specifies the plotly graph object for plotting
    the histogram (y) of the number of labels per message (x)
    """

    graph_dict = {
        "data": [Bar(x=x, y=y)],
        "layout": {
            "title": "Number of labels per message",
            "autosize": False,
            "height": 700,
            "width": 500,
            "yaxis": {
                "automargin": True,
                "title": "Number of labels",
            },
            "xaxis": {"title": "Number of messages"},
        },
    }
    return graph_dict


def plot_co_occurence_heatmap(x, y, z):
    """
    Returns a dictionary that specifies the plotly graph object for plotting
    a heatmap that informs of the pair-wise co-occurrence of labels
    """

    graph_dict = {
        "data": [Heatmap(x=x, y=y, z=z, hoverongaps=False, colorscale="Viridis")],
        "layout": {
            "title": "Co-occurrence of message labels",
            "autosize": False,
            "height": 800,
            "width": 1000,
            "yaxis": {
                "automargin": True,
                "autorange": "reversed",
            },
            "xaxis": {"tickangle": 60, "automargin": True},
        },
    }
    return graph_dict


##################################################################################################
# create list of ploty graph objects
##################################################################################################
df_labels = get_plot_dataframe(DATABASE_PATH)

label_names: Sequence[str] = [
    label.replace("_", " ") for label in df_labels.columns.values
]
counts_per_label: np.ndarray = df_labels.sum().values  # Fig. 1 Txt messages per label
hist_labels_per_msg: pd.Series = df_labels.sum(
    axis=1
).value_counts()  # Fig. 2 Labels per message
cooc_matrix = get_cooc_matrix(df_labels)  # Fig. 3 Co-occurence heatmap

graphs = [
    plot_counts_per_label(counts_per_label, label_names),
    plot_freq_of_messages_per_number_of_labels(
        hist_labels_per_msg.index, hist_labels_per_msg.values
    ),
    plot_co_occurence_heatmap(label_names, label_names, cooc_matrix),
]
##################################################################################################
