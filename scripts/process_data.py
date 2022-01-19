import argparse
from os import PathLike
import sys
from typing import Union
import pandas as pd
from sqlalchemy import create_engine, engine


def load_data(
    messages_filepath: Union[str, PathLike], categories_filepath: Union[str, PathLike]
) -> pd.DataFrame:
    """Merge inputs and labels from two separeted csv files into a DataFrame object.

    Args:
        messages_filepath: Raw csv file with text messages (inputs).
        categories_filepath: Raw csv file with class labels.

    Returns:
        df: Dataframe containing the merged text messages and class labels.
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, left_on="id", right_on="id", how="left").drop(
        "id", axis=1
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the merged dataset using functionality from pandas and save the results into another DataFrame.

    Args:
        df: Dataframe containing the unprocessed dataset.

    Returns:
        df_cleaned: Dataframe containing the cleaned dataset.
    """

    category_colnames = df["categories"].iloc[0].split(";")
    category_colnames = [item.split("-")[0] for item in category_colnames]
    categories = df["categories"].str.split(";", expand=True)
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(columns=["categories"], inplace=True)

    # add cleaned category cols
    df = pd.merge(df, categories, left_index=True, right_index=True)

    # drop duplicates
    df = df.drop_duplicates("message").reset_index(drop=True)
    return df


def save_data(
    df: pd.DataFrame, database_filepath: Union[str, PathLike] = "disaster_response.db"
) -> None:
    """Saves the cleaned dataset in a SQLite database file.

    Args:
        df: DataFrame containing the cleaned data.
        database_filepath (default: "disaster_response.db): Path to the SQLite file where the cleaned data shall be saved.
    """

    engine = create_engine("sqlite:///" + database_filepath)
    table_name = database_filepath.split(".db")[0].split("/")[-1]
    df.to_sql(table_name, engine, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments into a `argparse.Namespace` object.

    Returns:
        argparse.Namespace: An object whose attributes are the parsed command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x",
        "--inputs",
        type=str,
        default="data/disaster_messages.csv",
        help="Path to csv file containing text messages",
    )

    parser.add_argument(
        "-y",
        "--labels",
        type=str,
        default="data/disaster_categories.csv",
        help="Path to csv file containing class labels",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/disaster_response.db",
        help="Path to SQLite database file for writing the cleaned data",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    messages_filepath = args.inputs
    categories_filepath = args.labels
    database_filepath = args.output

    print(
        "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
            messages_filepath, categories_filepath
        )
    )
    df = load_data(messages_filepath, categories_filepath)

    print("Cleaning data...")
    df = clean_data(df)

    print("Saving data...\n    DATABASE: {}".format(database_filepath))
    save_data(df, database_filepath)

    print("Cleaned data saved to database!")


if __name__ == "__main__":
    main(parse_args())
