from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow import DAG
from datetime import datetime

master_dag = DAG(
    'master_dag',
    schedule_interval='0 0 * * *',
    start_date=datetime(2023, 8, 9),
    catchup=False,
)

trigger_data_ingestion = TriggerDagRunOperator(
    task_id='trigger_data_ingestion',
    trigger_dag_id='data_ingestion_dag',
    dag=master_dag,
)

trigger_data_processing = TriggerDagRunOperator(
    task_id='trigger_data_processing',
    trigger_dag_id='data_processing_dag',
    dag=master_dag,
)

trigger_feature_engineering = TriggerDagRunOperator(
    task_id='trigger_feature_engineering',
    trigger_dag_id='feature_engineering_dag',
    dag=master_dag,
)

trigger_predictions = TriggerDagRunOperator(
    task_id='trigger_predictions',
    trigger_dag_id='predictions_dag',
    dag=master_dag,
)

trigger_data_ingestion >> trigger_data_processing >> trigger_feature_engineering >> trigger_predictions
