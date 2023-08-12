from datetime import timedelta, datetime

from textwrap import dedent
import sys
import os

# Add the parent directory of the 'scripts' folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from airflow import DAG

from airflow.operators.python import PythonOperator

from scripts.data_processing import (db_ingestion, join_data_one_day_per_row,
                                     lagging_features, processing_df_to_db)
from scripts.data_ingestion import (get_news_data, get_historical_btc_data,
                                    merge_data, extract_content, df_to_db,
                                    get_latest_date_from_db, format_date_for_api)
from scripts.feature_engineering import (db_ingestion, preprocess_text,
                                         compute_tfidf, add_sentiment_and_tfidf_to_df,
                                         processing_df_to_db)

from scripts.config import db_path, table0, table1, table2, api_key, cryptobert_url, hugging_face_token


# Data Ingestion
def data_ingestion_master(api, database_name, tablename, **kwargs):
    latest_date = get_latest_date_from_db(database_name, tablename)
    formatted_latest_date = format_date_for_api(latest_date)
    news_data = get_news_data(api, formatted_latest_date)
    btc_data = get_historical_btc_data(api)
    merged_df = merge_data(btc_data, news_data)
    all_raw_data = extract_content(merged_df)
    df_to_db(all_raw_data, database_name, tablename)
    return print(f'Ingestion Master Executed check {tablename} in SQLite')


def data_processing_master(database_name, raw_table, processed_table, **kwargs):
    df = db_ingestion(database_name, raw_table)
    df = join_data_one_day_per_row(df)
    df = lagging_features(df)
    processing_df_to_db(df, database_name, processed_table)
    return print(f'Data Processing Master Executed check {processed_table} in SQLite')


def feature_eng_master(database_name, table_name1, table_name2, hf_endpoint, hf_token, **kwargs):
    df = db_ingestion(database_name, table_name1)
    processed_text_df = preprocess_text(df)
    tfidf_df = compute_tfidf(processed_text_df)
    merged_df = add_sentiment_and_tfidf_to_df(processed_text_df, tfidf_df, hf_endpoint, hf_token)
    processing_df_to_db(merged_df, database_name, table_name2)
    return print('Feature Engineering Master Executed! Check SQLite for the updates.')


# Data Ingestion Dag
with DAG(
        'data_pipeline_dag',
        default_args={
            'retries': 2,
            'retry_delay': timedelta(minutes=20)},
        description='DAG for data pipeline',
        schedule=timedelta(days=1),
        start_date=datetime(2023, 8, 13, 23, 55),
        catchup=False,
        tags=['Data Pipe']
) as data_pipeline:
    data_ingestion_task = PythonOperator(
        task_id='ingestion_task',
        python_callable=data_ingestion_master,
        op_args=[api_key, db_path, table0],
    )

    data_ingestion_task.doc_md = dedent(
        """Data Ingestion. See airflow/dags/dags.py data_ingestion_master for details on the task""")

    data_processing_task = PythonOperator(
        task_id='data_processing_task',
        python_callable=data_processing_master,
        op_args=[db_path, table0, table1],
    )

    data_processing_task.doc_md = dedent(
        """Data Processing. See airflow/dags/dags.py data_processing_master for details on the task""")

    feature_engineering_task = PythonOperator(
        task_id='feature_engineering_task',
        python_callable=feature_eng_master,
        op_args=[db_path, table1, table2, cryptobert_url, hugging_face_token],
    )

    feature_engineering_task.doc_md = dedent(
        """Feature Engineering. See airflow/dags/dags.py feature_engineering_master for details on the task""")

data_ingestion_task.set_downstream(data_processing_task)
data_processing_task.set_downstream(feature_engineering_task)