from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from model_dev import db_ingestion, add_classifier_predictions, db_name, table_name, clf_path
from joblib import load
import pandas as pd
from config import db_name, feature_engineered_df, clf_path, reg_path
import sqlite3

# Define the DAG
default_args = {
    'owner': 'you',
    'start_date': datetime(2023, 8, 9),
    'retries': 1,
}

dag = DAG('predictions_dag',
          default_args=default_args,
          schedule_interval='@daily')


def db_query(database_name, table_name):
    """
    This table name will be most recently pushed from feature eng dag
    """
    conn = sqlite3.connect(database_name)
    print('opened sqlite connection')
    query = f'SELECT * FROM {table_name}'
    df = pd.read_sql_query(query, conn)
    conn.close()
    print('closed sqlite connection')
    return df


def predictions_master(clf_model_path, reg_model_path, database_name, table):
    df = db_query(database_name, table)
    clf_feature_space = add_classifier_predictions(df, clf_model_path)
    reg_model = load(reg_model_path)
    reg_predictions = reg_model.predict([clf_feature_space])
    return reg_predictions[0]


