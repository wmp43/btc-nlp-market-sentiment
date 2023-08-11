from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from config import db_name, table1, table2, cryptobert_url, hugging_face_token




