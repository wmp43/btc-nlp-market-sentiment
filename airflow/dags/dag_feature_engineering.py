from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from config import db_name, table1, table2, cryptobert_url, hugging_face_token
from scripts.feature_engineering import (db_ingestion, preprocess_text, compute_tfidf,
                                         add_sentiment_and_tfidf_to_df, processing_df_to_db)


def feature_eng_master(database_name, table_name1, table_name2, hf_endpoint, hf_token, **kwargs):
    df = db_ingestion(database_name, table_name1)

    processed_text_df = preprocess_text(df)
    tfidf_df = compute_tfidf(processed_text_df)
    merged_df = add_sentiment_and_tfidf_to_df(processed_text_df, tfidf_df, hf_endpoint, hf_token)

    processing_df_to_db(merged_df, database_name, table_name2)

    return print('Feature Engineering Master Executed! Check SQLite for the updates.')


dag = DAG(
    'feature_engineering_dag',
    description='DAG for feature engineering',
    schedule_interval='0 0 * * *',
    start_date=datetime(2023, 8, 9),
    catchup=False,
    default_args={'retries': 1, 'retry_delay': timedelta(minutes=20)},
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering_task',
    python_callable=feature_eng_master,
    op_args=[db_name, table1, table2, cryptobert_url, hugging_face_token],
    dag=dag,
)
