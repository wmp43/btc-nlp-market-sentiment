from datetime import timedelta, datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from scripts.data_ingestion import (get_news_data,
                                    get_historical_btc_data, merge_data, extract_content, df_to_db, get_latest_date_from_db,
                                    format_date_for_api)
from config import api_key, db_name, table0


def ingestion_master(api, database_name, tablename, **kwargs):
    latest_date = get_latest_date_from_db(database_name, tablename)
    formatted_latest_date = format_date_for_api(latest_date)
    news_data = get_news_data(api, formatted_latest_date)
    btc_data = get_historical_btc_data(api)
    merged_df = merge_data(btc_data, news_data)
    all_raw_data = extract_content(merged_df)
    df_to_db(all_raw_data, database_name, tablename)
    return print(f'Ingestion Master Executed check {tablename} in SQLite')


dag = DAG(
    'data_ingestion_dag',
    description='DAG for data ingestion',
    schedule_interval='0 0 * * *',
    start_date=datetime(2023, 8, 9),
    catchup=False,
    default_args={'retries': 1, 'retry_delay': timedelta(minutes=20)},
)

ingestion_task = PythonOperator(
    task_id='ingestion_task',
    python_callable=ingestion_master,
    op_args=[api_key, db_name, table0],
    dag=dag,
)

