"""Contains utility functions for data and model loading."""

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


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
