# Disaster Response Pipeline Project

This is the Disaster Response Pipeline project for the Data Science
Nanodegree from Udacity. The project provides an overview of ETL and
machine learning pipelines.

The project uses real messages sent during disaster events. The goal
is to create a machine learning pipeline that categorizes these events
in a way that new messages sent can be relayed to the appropriate
relief agency.

In order to install all Python the dependencies, please execute the
following command.

```
$ pip install -r requirements.txt
```

## ETL Pipeline

The ETL process merges data from two different files that include the
messages and their categories. It also cleans the data, removes
duplicates and stores this data in a SQLite data base. To execute this
process, run the following command:

```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

## ML Pipeline

The ML pipeline uses the data stored in the SQLite database and splits
it into training and test data. It prepares the messages data for
training. It trains and tunes the model and exports the final model as
a pickle file. To run the pipeline, execute the command below.

```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

## Web Application

Finally, the project implements a simple web application that provides
a few plots describing the training dataset. It also provides a
platform to classify messages on the categories used for training. To run the application, execute the following command:

```
python app/run.py
```

Visit the project [url]( http://0.0.0.0:3001/)
