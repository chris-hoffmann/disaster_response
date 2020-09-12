import sys
import time
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, engine
from app.utils import tokenize, clean_further

from sklearn.metrics import classification_report, f1_score, make_scorer\
    , multilabel_confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def load_data(database_filepath):
    '''
    Load the database file containing the pre-cleaned data
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filepath.replace('/', '.').split('.')[-2]
    
    df = pd.read_sql_table(table_name, engine)
    df = clean_further(df)
    
    category_names = df.iloc[:,4:].columns.tolist()
    X = df.message.values
    y = df.iloc[:,4:].values
 
    return X, y, category_names


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(random_state=42, bootstrap=False,
                                       class_weight='balanced'))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [3, 4, 5]
    }

    f1_micro = make_scorer(f1_score, average='micro')
    cv = RandomizedSearchCV(pipeline, param_distributions=parameters,
                            scoring=f1_micro, cv=3, n_jobs=-1, verbose=2,
                            n_iter=100)
  
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    cls_report = classification_report(y_test, y_pred,
                                       target_names=category_names,
                                       output_dict=True)
    with open('./models/cls_report.pkl', 'wb') as handle:
        pickle.dump(cls_report, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(classification_report(y_test, y_pred, target_names=category_names))
    confusion_array = multilabel_confusion_matrix(y_test, y_pred)
    np.save('./models/confusion_matrix.npy', confusion_array)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Tuning model via random grid search...')
        start = time.time()
        model.fit(X_train, y_train)
        stop = time.time()
        print(f"Training time: {(stop - start)/60.} min")

        print('Best parameters...')
        print(model.best_params_)

        print('Evaluate model...')
        evaluate_model(model, X_test, y_test, category_names)
     
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument\nand the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample:\n'
              'python ./code/train_classifier.py ./data/disaster_response.db '\
              './models/classifier.pkl')


if __name__ == '__main__':
    main()
