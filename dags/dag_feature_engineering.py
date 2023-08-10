from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from feature_engineering import feature_eng_pipeline
from config import db_name, processed_table_name, table_name, cryptobert_url, hugging_face_token


def feature_engineering_master(**kwargs):
    feature_eng_pipeline(db_name, processed_table_name, table_name, cryptobert_url, hugging_face_token)
    return print('Feature Engineering Master Executed! Check SQLite for the updates.')


dag = DAG(
    'feature_engineering_dag',
    description='DAG for feature engineering',
    schedule_interval='@daily',
    start_date=datetime(2023, 8, 9),
    catchup=False,
    default_args={'retries': 1, 'retry_delay': timedelta(minutes=20)},
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering_task',
    python_callable=feature_engineering_master,
    dag=dag,
)
