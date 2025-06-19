from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import subprocess
import json
import logging

CONTAINER = "data_ml_container_asm2"
BASELINE  = "data/baseline_features.csv"
FEATURES  = "data/feature_clickstream.csv"
CAND_DIR  = "model_bank/"
PROD_DIR  = "model_bank/production/"
LABEL_DIR = "datamart/gold/label_store/"
REPORT_DIR = "reports/"  

def infer_pipeline(snapshot: str) -> str:
    cmd = [
        "docker", "exec", CONTAINER, "python", "run_inference_and_monitor_pipeline.py",
        "--snapshotdate", snapshot,
        "--baselinepath", BASELINE,
        "--featurepath",  FEATURES,
        "--modeldir",     PROD_DIR,
        "--quiet",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = out.strip().split("\n")[-1]  # keep last line only
    logging.info("Inference payload: %s", payload)
    return payload


def drift_decision(**context):
    """BranchPythonOperator callable – decides which path to follow."""
    payload_raw = context["ti"].xcom_pull(task_ids="run_infer_monitor")
    try:
        payload = json.loads(payload_raw or "{}")
        psi_max = payload.get("psi_max", 1.0)
    except Exception:
        logging.error("Could not parse inference payload – forcing retrain")
        return "trigger_retrain"

    snapshot = context["ds"]
    logging.info("Snapshot %s – max PSI = %.4f", snapshot, psi_max)
    return "continue_pipeline" if psi_max < 0.20 else "trigger_retrain"


def _noop():
    logging.info("PSI below threshold – pipeline continues.")

with DAG(
    dag_id="run_data_and_ml_pipeline",
    start_date=datetime(2025, 6, 1),
    catchup=False,
    schedule="@monthly",
    tags=["ml", "training", "inference"],
    description="ETL + Training + Inference with automatic drift retrain",
) as dag:

    run_bronze = BashOperator(
        task_id="run_bronze_data_pipeline",
        bash_command=f"docker exec {CONTAINER} python run_bronze_data_pipeline.py",
    )
    run_silver = BashOperator(
        task_id="run_silver_data_pipeline",
        bash_command=f"docker exec {CONTAINER} python run_silver_data_pipeline.py",
    )
    run_gold = BashOperator(
        task_id="run_gold_data_pipeline",
        bash_command=f"docker exec {CONTAINER} python run_gold_data_pipeline.py",
    )
    extract_base = BashOperator(
        task_id="run_extract_baseline",
        bash_command=(
            f"docker exec {CONTAINER} python run_extract_baseline_pipeline.py "
            f"--baselinepath {BASELINE}"
        ),
    )

    train_lr = BashOperator(
        task_id="train_logistic_regression",
        bash_command=(
            f"docker exec {CONTAINER} python train_model.py "
            "--modeltype logistic_regression --snapshotdate {{ ds }} "
            f"--featurepath {FEATURES} --labeldir {LABEL_DIR} --modeldir {CAND_DIR}"
        ),
    )

    train_xgb = BashOperator(
        task_id="train_xgboost",
        bash_command=(
            f"docker exec {CONTAINER} python train_model.py "
            "--modeltype xgboost --snapshotdate {{ ds }} "
            f"--featurepath {FEATURES} --labeldir {LABEL_DIR} --modeldir {CAND_DIR}"
        ),
    )

    select_best = BashOperator(
        task_id="select_best_model",
        bash_command=(
            f"docker exec {CONTAINER} python select_model.py "
            f"--candidatedir {CAND_DIR} --productiondir {PROD_DIR} "
            "--startdate {{ ds }} --enddate {{ ds }}"
        ),
    )

    run_infer_monitor = PythonOperator(
        task_id="run_infer_monitor",
        python_callable=infer_pipeline,
        op_args=["{{ ds }}"],
        do_xcom_push=True,
    )

    drift_branch = BranchPythonOperator(
        task_id="drift_branch",
        python_callable=drift_decision,
    )

    continue_pipeline = PythonOperator(
        task_id="continue_pipeline",
        python_callable=_noop,
    )

    export_metrics = BashOperator(
        task_id="export_metrics",
        bash_command=(
            f"docker exec {CONTAINER} python export_run_metrics.py "
            f"--outdir {REPORT_DIR}{{{{ ds }}}}"                 # ← changed
        ),
    )

    build_report = BashOperator(
        task_id="build_report",
        bash_command=(
            f"docker exec {CONTAINER} python generate_performance_report.py "
            f"--metrics {REPORT_DIR}{{{{ ds }}}}/metrics.parquet "
            f"--outdir  {REPORT_DIR}{{{{ ds }}}}"
        ),
    )
    
    alert_drift = BashOperator(
        task_id="alert_drift",
        bash_command=(
            f"docker exec -e AIRFLOW_API=http://airflow-apiserver:8080/api/v1/dags "
            f"{CONTAINER} python alert_drift.py --retrain-dag run_data_and_ml_pipeline"
        ),
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="run_data_and_ml_pipeline",
        conf={"snapshot": "{{ ds }}"},
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    run_bronze >> run_silver >> run_gold >> extract_base \
        >> [train_lr, train_xgb] >> select_best \
        >> run_infer_monitor >> drift_branch

    drift_branch >> continue_pipeline >> export_metrics >> build_report >> alert_drift

    drift_branch >> trigger_retrain
