from datetime import timedelta, datetime

from textwrap import dedent
import sys
import os

# Add the parent directory of the 'scripts' folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from airflow import DAG

from airflow.operators.python import PythonOperator
from airflow.decorators import dag, task

from scripts.data_processing import (db_ingestion, join_data_one_day_per_row,
                                     lagging_features, processing_df_to_db)
from scripts.data_ingestion import (get_news_data, get_historical_btc_data,
                                    merge_data, extract_content, df_to_db,
                                    get_latest_date_from_db, format_date_for_api)
from scripts.feature_engineering import (db_ingestion, preprocess_text,
                                         compute_tfidf, add_sentiment_and_tfidf_to_df,
                                         processing_df_to_db)

from scripts.config import db_path, table0, table1, table2, api_key, cryptobert_url, hugging_face_token


# https://airflow.apache.org/docs/apache-airflow/2.6.1/tutorial/pipeline.html

# Data Ingestion


@task
def data_ingestion_master(api, database_name, tablename, **kwargs):
    latest_date = get_latest_date_from_db(database_name, tablename)
    formatted_latest_date = format_date_for_api(latest_date)
    news_data = get_news_data(api, formatted_latest_date)
    btc_data = get_historical_btc_data(api)
    merged_df = merge_data(btc_data, news_data)
    all_raw_data = extract_content(merged_df)
    df_to_db(all_raw_data, database_name, tablename)
    return print(f'Ingestion Master Executed check {tablename} in SQLite')


@task
def data_processing_master(database_name, raw_table, processed_table, **kwargs):
    df = db_ingestion(database_name, raw_table)
    df = join_data_one_day_per_row(df)
    df = lagging_features(df)
    processing_df_to_db(df, database_name, processed_table)
    return print(f'Data Processing Master Executed check {processed_table} in SQLite')


@task
def feature_eng_master(database_name, table_name1, table_name2, hf_endpoint, hf_token, **kwargs):
    df = db_ingestion(database_name, table_name1)
    processed_text_df = preprocess_text(df)
    tfidf_df = compute_tfidf(processed_text_df)
    merged_df = add_sentiment_and_tfidf_to_df(processed_text_df, tfidf_df, hf_endpoint, hf_token)
    processing_df_to_db(merged_df, database_name, table_name2)
    return print('Feature Engineering Master Executed! Check SQLite for the updates.')


@dag(
    dag_id='btc_text_data_pipeline',
    default_args={
             'retries': 2,
             'retry_delay': timedelta(minutes=7)},
    schedule=timedelta(days=1),  # Or your preferred schedule
    start_date=datetime(2023, 8, 27, 23, 55),
    catchup=False,
    dagrun_timeout=timedelta(minutes=60)
)
def dag_func():
    data_ingestion = data_ingestion_master(api_key, db_path, table0)
    data_processing = data_processing_master(db_path, table0, table1)
    feature_eng = feature_eng_master(db_path, table1, table2, cryptobert_url, hugging_face_token)

    data_ingestion >> data_processing >> feature_eng

# Initialize the DAG
dag_instance = dag_func()




