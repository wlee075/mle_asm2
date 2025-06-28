from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python import BranchPythonOperator
from datetime import datetime, timedelta
import subprocess
import os

CONTAINER = "data_ml_container_asm2"
BASELINE  = "data/baseline_features.csv"
MODEL_BANK = "model_bank/"

default_args = {
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def _branch_for_training():
    cmd = ["docker", "exec", CONTAINER, "bash", "-lc", "ls model_bank/*.pkl"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return "trigger_inference"
    else:
        return "trigger_training"

with DAG(
    dag_id="etl_pipeline",
    default_args=default_args,
    schedule="@monthly",
    catchup=True,
    tags=["etl"],
) as dag:

    run_bronze = BashOperator(
        task_id="run_bronze_data_pipeline",
        bash_command=f"docker exec {CONTAINER} python pipelines/run_bronze_data_pipeline.py",
    )
    run_silver = BashOperator(
        task_id="run_silver_data_pipeline",
        bash_command=f"docker exec {CONTAINER} python pipelines/run_silver_data_pipeline.py",
    )
    run_gold = BashOperator(
        task_id="run_gold_data_pipeline",
        bash_command=f"docker exec {CONTAINER} python pipelines/run_gold_data_pipeline.py",
    )
    extract_base = BashOperator(
        task_id="run_extract_baseline",
        bash_command=(
            f"docker exec {CONTAINER} python pipelines/run_extract_baseline_pipeline.py "
            f"--baselinepath {BASELINE}"
        ),
    )

    decide_train = BranchPythonOperator(
        task_id="decide_train",
        python_callable=_branch_for_training,
    )

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="train_ml_pipeline",
        conf={"snapshot": "{{ ds }}"},
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    trigger_inference = TriggerDagRunOperator(
        task_id="trigger_inference",
        trigger_dag_id="inference_pipeline",
        conf={
          "snapshot":   "{{ ds }}",
          "combinedpath": "data/gold/combined_labeled.parquet"
        },
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    run_bronze >> run_silver >> run_gold >> extract_base >> decide_train >> [trigger_training, trigger_inference]