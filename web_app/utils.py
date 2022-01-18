"""Contains utility functions for data and model loading as well as for
generating graph objects, which are then plotted in the web app using Plotly"""

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from plotly.graph_objs import Bar, Heatmap


def get_cooc_matrix(df_labels: pd.DataFrame) -> np.ndarray:
    """Returns the symmetric pair-wise co-occurrence matrix.

    Args:
        df_label: DataFrame with columns corresponding to class labels.

    Returns:
        cooc_matrix: 2D np.array representing the co-occurrence heatmap of class labels. The shape is $k \times k$, where $k$ refers to the number of clases.
    """
    k = df_labels.shape[-1]
    co_occ_matrix = np.zeros((k, k))
    for i, la in enumerate(df_labels.columns.values):
        df_temp = df_labels.loc[df_labels[la] == 1]
        n, _ = df_temp.shape
        co_occ_matrix[i, :] = df_temp.sum(axis=0).values / n
    return co_occ_matrix


def tokenize(text):
    """
    Tokenizes a text message and transforms the resulting tokens to their base
    form using the WordNet lemmatizer
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def filter_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pandas dataframe that lacks the `child_alone` column and rows
    for which the uniformative `related` column is equal to 1.
    """
    df = df.drop(columns=["child_alone"])
    df = df[df.related == 1]
    return df


def plot_counts_per_label(x, y):
    """
    Returns a list that contains a dictionary to specify a plotly graph
    object for plotting the number of messages (y) per label (x) as a
    horizontal bar plot using Plotly
    """
    graph = [
        {
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
    ]
    return graph


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
