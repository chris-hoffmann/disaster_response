import sys
import pandas as pd
from sqlalchemy import create_engine, engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load and merge both csv files
    '''
    messages = pd.read_csv(messages_filepath)
    categories= pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, left_on='id', right_on='id',
                    how='left').drop('id', axis=1)

def clean_data(df):
    '''
    Clean the merged dataset using Pandas
    '''
    # clean up df['categories']
    category_colnames = df['categories'].iloc[0].split(';')
    category_colnames = [item.split('-')[0] for item in category_colnames]
    categories = df['categories'].str.split(';',expand=True) 
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int) 

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # add cleaned category cols
    df = pd.merge(df, categories, left_index=True, right_index=True)
     
    # drop duplicates
    df = df.drop_duplicates('message').reset_index(drop=True)
    return df

def save_data(df, database_filepath):
    '''
    Save the cleaned dataset in a SQLite database file
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filepath.split('.db')[0].split('/')[-1]
    df.to_sql(table_name, engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, \
        database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the file-paths of the messages and categories'\
              ' datasets as the first and second argument respectively, \n'
              'as well as the filepath of the database to save the cleaned '
              'data to as the third argument. \n\nExample:\npython ./code/'\
              'process_data.py ./data/disaster_messages.csv ./data/'\
              'disaster_categories.csv ./data/disaster_response.db')


if __name__ == '__main__':
    main()
