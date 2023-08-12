from datetime import timedelta, datetime

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

from scripts.config import db_name, table0, table1, table2, api_key, cryptobert_url, hugging_face_token


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


# Data Ingestion Dag
with DAG(
        'data_ingestion_dag',
        description='DAG for data ingestion',
        schedule_interval='0 0 * * *',
        start_date=datetime(2023, 8, 12),
        catchup=False,
        default_args={'retries': 1, 'retry_delay': timedelta(minutes=20)},
) as data_ingestion_dag:
    data_ingestion_task = PythonOperator(
        task_id='ingestion_task',
        python_callable=data_ingestion_master,
        op_args=[api_key, db_name, table0],
        dag=data_ingestion_dag,
    )


# Data Processing
def data_processing_master(database_name, raw_table, processed_table, **kwargs):
    df = db_ingestion(database_name, raw_table)
    df = join_data_one_day_per_row(df)
    df = lagging_features(df)
    processing_df_to_db(df, database_name, processed_table)
    return print(f'Data Processing Master Executed check {processed_table} in SQLite')


# Data Processing Dag
with DAG(
        'data_processing_dag',
        description='DAG for data processing',
        schedule_interval='0 0 * * *',
        start_date=datetime(2023, 8, 12),  # Set the start date appropriately
        catchup=False,
        default_args={'retries': 1, 'retry_delay': timedelta(minutes=20)},
) as data_processing_dag:
    data_processing_task = PythonOperator(
        task_id='data_processing_task',
        python_callable=data_processing_master,
        op_args=[db_name, table0, table1],
        dag=data_processing_dag,
    )


# Feature Engineering
def feature_eng_master(database_name, table_name1, table_name2, hf_endpoint, hf_token, **kwargs):
    df = db_ingestion(database_name, table_name1)
    processed_text_df = preprocess_text(df)
    tfidf_df = compute_tfidf(processed_text_df)
    merged_df = add_sentiment_and_tfidf_to_df(processed_text_df, tfidf_df, hf_endpoint, hf_token)
    processing_df_to_db(merged_df, database_name, table_name2)
    return print('Feature Engineering Master Executed! Check SQLite for the updates.')


# Feature Engineering DAG
with DAG(
        'feature_engineering_dag',
        description='DAG for feature engineering',
        schedule_interval='0 0 * * *',
        start_date=datetime(2023, 8, 9),
        catchup=False,
        default_args={'retries': 1, 'retry_delay': timedelta(minutes=20)},
) as feature_eng_dag:
    feature_engineering_task = PythonOperator(
        task_id='feature_engineering_task',
        python_callable=feature_eng_master,
        op_args=[db_name, table1, table2, cryptobert_url, hugging_face_token],
        dag=feature_eng_dag,
    )

# Task Pipeline and definitions
data_ingestion_task >> data_processing_task >> feature_engineering_task
