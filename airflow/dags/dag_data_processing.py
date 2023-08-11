from datetime import timedelta, datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from scripts.data_processing import (
    db_ingestion, join_data_one_day_per_row,
    lagging_features, processing_df_to_db
)
from config import db_name, table0, table1


def data_processing_master(database_name, raw_table, processed_table, **kwargs):
    df = db_ingestion(database_name, raw_table)
    df = join_data_one_day_per_row(df)
    df = lagging_features(df)
    processing_df_to_db(df, database_name, processed_table)
    return print(f'Data Processing Master Executed check {processed_table} in SQLite')


dag = DAG(
    'data_processing_dag',
    description='DAG for data processing',
    schedule_interval='0 0 * * *',
    start_date=datetime(2023, 8, 10),  # Set the start date appropriately
    catchup=False,
    default_args={'retries': 1, 'retry_delay': timedelta(minutes=20)},
)

data_processing_task = PythonOperator(
    task_id='data_processing_task',
    python_callable=data_processing_master,
    op_args=[db_name, table0, table1],
    dag=dag,
)
