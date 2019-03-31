#! /usr/bin/env python3

import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """
    Load two csv files and merge them on column `id`
    Inputs:
    - messages_filepath: location of the messages csv
    - categories_filepath: location of the categories csv

    Output:
    - merged dataframe with data from both csv files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on=['id'])


def clean_data(df):
    """
    Clean the dataframe by moving the data in the category
    column to multiple columns. It also drops duplicates.
    Inputs:
    - df: dataframe

    Output:
    - cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    categories.columns = [x[0] for x in row.str.split('-')]
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.replace(column+'-', '').astype(str)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        categories[column] = categories[column].apply(lambda x: x if x < 2 else 1)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Save the data into a sqlite database
    Inputs:
    - df: dataframe
    - database_filename: location of the database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster-response', engine, index=False)


def main():
    """
    Execute the ETL pipeline
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
