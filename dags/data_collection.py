from airflow import DAG
from airflow.operators.python import PythonOperator
from data_ingestion import (parse_float_date, parse_date, get_news_data,
                            get_historical_btc_data, merge_data, extract_text,
                            extract_content, df_to_db)
from config import API_KEY, TIME_FROM, DB_NAME, RAW_TABLE_NAME

def ingestion_master(api_key, time_from, db_name, raw_table_name):
    news_data = get_news_data(api_key, time_from)    
    btc_data = get_historical_btc_data(api_key)
    merged_df = merge_data(btc_data, news_data)
    all_raw_data = extract_content(merged_df)
    df_to_db(all_raw_data, db_name, raw_table_name)
    return print('check SQLite!')
