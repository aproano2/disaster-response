#! /usr/bin/env python3

import sys
import re
import pandas as pd
import numpy as np
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# download necessary NLTK data
nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """
    Load data from sqlite database and return data for
    modeling
    Input:
    - database_filepath: location of the database
    
    Output:
    - X: training data
    - y: taining labels
    - category names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster-response',engine)
    X = df['message']
    y = df.iloc[:,4:]
    return X, y, y.columns

def tokenize(text):
    """
    Tokenize the text used for training. It eliminates urls, tokenizes, and
    lemmatizes the text. It does not consider stop words.
    Inputs:
    - text
    
    Outputs:
    - tokenized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(tok.strip().lower()) for tok in tokens
                    if tok.strip().lower() not in stop_words]


def build_model():
    """
    Build and optimize the model

    Output:
    - optimal model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2']
    }

    return GridSearchCV(pipeline, param_grid=parameters,
                        cv=2, verbose=1)


def display_results(y_test, y_pred, category_names):
    """
    Display the results from model tesitng. It prints F1-score, precision, 
    recall and accuracy.
    Input:
    - y_test: training labels
    - y_pred: predictions
    - category_names
    """
    for n in range(len(category_names)):
        accuracy = (y_pred[:, n] == y_test.values[:, n]).mean()
        print("Label:", category_names[n])
        print("F1-score:", f1_score(y_test.values[:, n], y_pred[:, n]))
        print("Precision:", precision_score(y_test.values[:, n], y_pred[:, n]))
        print("Recall:", recall_score(y_test.values[:, n], y_pred[:, n]))
        print("Accuracy:", accuracy)
        print()  

        
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model.
    Input:
    - model
    - X_test: data used for testing
    - Y_test: test labels
    - category_names
    """
    y_pred = model.predict(X_test)
    display_results(Y_test, y_pred, category_names)


def save_model(model, model_filepath):
    """
    Save model
    Input:
    - model
    - model_filepath: location for saving the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
