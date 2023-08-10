from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from data_ingestion import ingestion_master
from config import api_key, db_name, raw_table_name

default_args = {
    'owner': 'your_name',
    'start_date': datetime(2023, 6, 2),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_workflow_dag',
    default_args=default_args,
    description='DAG to update crypto news and price data',
    schedule_interval=timedelta(days=1),  # Adjust this based on how often you want to run it
    catchup=False,
)

ingest_task = PythonOperator(
    task_id='ingest_crypto_data',
    python_callable=ingestion_master,
    op_args=[api_key, db_name, raw_table_name],
    dag=dag,
)
