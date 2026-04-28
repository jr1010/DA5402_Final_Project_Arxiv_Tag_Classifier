# AIRFLOW DAG FOR DATA INGESTION AND CLEANINGimport sys
import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


# -------------------------
# Fix import path (project root)
# -------------------------
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

# Import task logic
from airflow.tasks.ingest import main as ingest_main
from airflow.tasks.clean import main as clean_main


# -------------------------
# Default args
# -------------------------
default_args = {
    "owner": "airflow",
    "retries": 2,
}


# -------------------------
# DAG definition
# -------------------------
with DAG(
    dag_id="arxiv_monthly_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="0 0 1 * *",   # run on 1st of every month
    catchup=False,
    max_active_runs=1,
    tags=["arxiv", "mlops"],
) as dag:

    # -------------------------
    # Ingest task
    # -------------------------
    def ingest_wrapper(execution_date, **kwargs):
        ingest_main(execution_date.strftime("%Y-%m-%d"))

    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_wrapper,
    )

    # -------------------------
    # Clean task
    # -------------------------
    clean_task = PythonOperator(
        task_id="clean_data",
        python_callable=clean_main,
    )

    # -------------------------
    # DVC pipeline task
    # -------------------------
    dvc_task = BashOperator(
        task_id="run_dvc_pipeline",
        bash_command="cd {{ dag_run.conf.get('repo_root', '.') }} && dvc repro",
    )

    # -------------------------
    # Dependencies
    # -------------------------
    ingest_task >> clean_task >> dvc_task