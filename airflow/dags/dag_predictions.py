from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from scripts.model_dev import add_classifier_predictions
from joblib import load
import pandas as pd
from config import db_name, clf_path, reg_path, table2
import sqlite3


def db_query(database_name, tablename):
    """
    This table name will be most recently pushed from feature eng dag
    """
    conn = sqlite3.connect(database_name)
    print('opened sqlite connection')
    query = f'SELECT * FROM {tablename}'
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


dag = DAG(
    'predictions_dag',
    description='DAG for predictions',
    schedule_interval='0 0 * * *',
    start_date=datetime(2023, 8, 9),
    catchup=False,
    default_args={'retries': 1, 'retry_delay': timedelta(minutes=20)}
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering_task',
    python_callable=predictions_master,
    op_args=[clf_path, reg_path, db_name, table2],
    dag=dag,
)
