import argparse
import datetime
import os
import pathlib
import pickle
import time
from enum import Enum
from tempfile import mkdtemp
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from web_app.utils import filter_labels, tokenize

Classifier = Union[Pipeline, BaseEstimator]


def load_data(database_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Returns inputs, labels and class-names from a database file that contains cleaned data.

    Args:
        database_file: Name of the SQLite database file including extension.

    Returns:
        3-element tuple containing:
        - X: Inputs as a numpy array of shape (N, D), where N is the number of inputs and D is the feature dimension.
        - y: Labels as a numpy array of shape (N, ).
        - class_names: List of class-names of len(K), where K is the number of classes in the database.
    """

    sql_path = "sqlite:///" + database_file
    # sql_path = "sqlite:///data/disaster_response.db"
    engine = create_engine(sql_path)
    table_name = database_file.replace("/", ".").split(".")[-2]
    # table_name = 'disaster_response'

    df = pd.read_sql_table(table_name, engine)
    df = filter_labels(df)

    class_names = df.iloc[:, 4:].columns.tolist()
    X = df.message.values
    y = df.iloc[:, 4:].values

    return X, y, class_names


def init_pipeline(
    clf: Classifier, clf_hparams: Optional[Dict], score_func: Any = "f1_weighted"
) -> Classifier:
    """Initialize a text classification model, based on tf-idf (term frequency inverse document frequency) features,
    for hyperparameter tuning via random search.

    Args:
        clf: Classifier
        clf_hparams (optional): Hyperparameters  for `clf`.
        score_func (default: "f1_weighted"): Scoring function for evaluating hyperparameter trials.

    Returns:
        pipeline: Text classification pipeline intialized for hyperparameter tuning via random search.

    """

    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=0)
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", clf),
        ],
        memory=memory,
    )

    hparams = {
        "vect__ngram_range": ((1, 1), (1, 2)),
        "vect__max_df": (0.5, 0.75, 1.0),
        "vect__max_features": (None, 5000, 10000),
        "tfidf__use_idf": (True, False),
    }

    if clf_hparams:
        hparams.update(clf_hparams)

    # use a `random_state: int` to allow for fold-wise comparison
    cv = RandomizedSearchCV(
        pipeline,
        param_distributions=hparams,
        scoring=score_func,
        cv=3,
        n_jobs=-1,
        verbose=2,
        n_iter=100,
        random_state=42,
    )

    return cv


def evaluate_model(
    model: Classifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    exp_folder: Union[str, pathlib.Path],
) -> None:
    """Evaluates the model performance on the test set and writes the outcome to a pickle (*.pkl) and
    numpy file (*.npy).

    The pickle file contains the classification report, which lists precision, recall and F1-score for each class
    and across the entire dataset. The numpy file contains the 2x2 confusion matrices for each class saved as an
    array of shape (K, 2, 2), where K refers to the number of classes.


    Args:
        model: Fitted scikit-learn estimator for text classification.
        X_test: Test inputs as a numpy array of shape (N, D), where N is the number of inputs and D is the feature dimension.
        y_test: Test labels as a numpy array of shape (N, ).
    """

    y_pred = model.predict(X_test)
    cls_report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    with open(os.path.join(exp_folder, "cls_report.pkl"), "wb") as handle:
        pickle.dump(cls_report, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(classification_report(y_test, y_pred, target_names=class_names))
    confusion_array = multilabel_confusion_matrix(y_test, y_pred)
    np.save(os.path.join(exp_folder, "confusion_matrices.npy"), confusion_array)


def save_model(
    model: Classifier, folder: Union[str, pathlib.Path], file_name: str
) -> None:
    """Writes a model object to a pickle (*.pkl) file.

    Args:
        model: Model object.
        folder: Parent directory in which the model is saved.
        file_name: Name of the pickle file including the final extension (*.pkl).
    """
    fpath = os.path.join(folder, file_name)
    pickle.dump(model, open(fpath, "wb"))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments into a `argparse.Namespace` object.

    Returns:
        argparse.Namespace: An object whose attributes are the parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        default="data/disaster_response.db",
        help="Path to database file",
        required=True,
    )
    parser.add_argument(
        "-clf",
        "--classifier",
        type=AvailableClassifier.from_string,
        choices=list(AvailableClassifier),
        default="logit",
        help="Select available classifier: logit or random_forest",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        # default="models/classifier.pkl",
        help="Path to the saved sklearn modelfor evaluation on the test set",
    )

    return parser.parse_args()


class AvailableClassifier(Enum):
    logit = (
        OneVsRestClassifier(LogisticRegression(class_weight="balanced", max_iter=250)),
        {"clf__estimator__C": (np.logspace(-4, 4, 10))},
    )
    random_forest = (
        RandomForestClassifier(class_weight="balanced", bootstrap=False),
        {"clf__n_estimators": [50, 100, 200], "clf__min_samples_split": [3, 4, 5]},
    )

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(string):
        try:
            return AvailableClassifier[string]
        except KeyError:
            raise ValueError()


def main(args: argparse.Namespace) -> None:
    rng = np.random.RandomState(0)

    database_filepath = args.database
    print("Loading data...\n    DATABASE: {}".format(database_filepath))
    X, y, class_names = load_data(database_filepath)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rng
    )

    if not args.model:
        # perform training if no model file is given
        exp_name = "_".join(
            [
                args.classifier._name_,
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            ]
        )
        exp_folder = os.path.join("experiments", exp_name)
        os.makedirs(exp_folder)

        clf, clf_hparams = args.classifier.value
        clf.random_state = rng

        print("Building text classification pipeline...")
        text_clf_pipeline = init_pipeline(clf=clf, clf_hparams=clf_hparams)

        print("Tune model via random hyperarameter search...")
        start = time.time()
        text_clf_pipeline.fit(X_train, y_train)
        print(f"Training time: {(time.time() - start)/60.} min")

        print(f"Best hyperparameters:\n{text_clf_pipeline.best_params_}")

        print("Saving model...\n    MODEL: {}_model.pkl".format(args.classifier._name_))
        save_model(
            text_clf_pipeline,
            folder=exp_folder,
            file_name=f"{args.classifier._name_}_model.pkl",
        )
        print("Trained model saved!")

    else:
        # load trained model for evaluation
        text_clf_pipeline = joblib.load(args.model)
        exp_folder = args.model.rsplit("/", maxsplit=1)[0]

    print("Evaluate model on test set...")
    evaluate_model(
        text_clf_pipeline, X_test, y_test, class_names, exp_folder=exp_folder
    )
    print("Evaluation results saved!")


if __name__ == "__main__":
    main(parse_args())
