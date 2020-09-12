import os
import json
import plotly
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine
from app.utils import tokenize, clean_further, plot_counts_per_label, \
plot_freq_of_messages_per_number_of_labels, plot_co_occurence_heatmap


app = Flask(__name__, template_folder=os.path.abspath('./app/templates'))

# load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# generate variables for Figure 1 'Number of messages per label'
df = clean_further(df)
df_labels = df.drop(df.columns[:4],axis=1)
labels = df_labels.columns.values
counts_per_label = df_labels.sum().values

# generate variables for Figure 2 'Histogram of the number of labels per \
# message'
labels_per_msg = df_labels.sum(axis=1)
hist_labels_per_msg = labels_per_msg.value_counts()

# generate variables for Figure 3 'Co-occurrence heatmap'
co_occ_matrix = np.zeros((34,34))
for i,lab in enumerate(labels):
    df_temp = df_labels.loc[df_labels[lab] == 1]
    n,_ = df_temp.shape
    co_occ_matrix[i,:] = df_temp.sum(axis=0).values / n
label_names = [lab.replace('_', ' ') for lab in labels]


@app.route('/')
@app.route('/index')
def index():
    '''
    Generates an index web page that displays plots summarizing the dataset
    and receives user input text for prediction
    '''

    # generate a list of plotly graph objects
    graphs = plot_counts_per_label(counts_per_label, label_names)
    graphs.append(plot_freq_of_messages_per_number_of_labels(
        hist_labels_per_msg.index, hist_labels_per_msg.values))
    graphs.append(plot_co_occurence_heatmap(label_names, label_names,
        co_occ_matrix))
    
    # encode plotly graph objects in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    '''
    Generates a web page that handles user query and displays prediction
    results
    '''

    # save user input in query
    query = request.args.get('query', '') 

    # use model to perform classification for query text
    classification_labels = model.predict([query])[0]
    	    
    # remove child_alone label!
    classification_results = dict(zip(df_labels.columns,
                                      classification_labels))

    # This will render the go.html - see template for further information.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


@app.route('/about/')
def about():
    '''
    Generates a web page that informs about the project and the underlying
    machine learning model
    '''
    return render_template('about_the_model.html')

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()