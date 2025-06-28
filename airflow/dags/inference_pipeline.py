from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import subprocess
import json
import logging
import os

CONTAINER   = "data_ml_container_asm2"
BASELINE    = "data/baseline_features.csv"
PROD_DIR    = "model_bank/production/"
REPORT_DIR  = "reports/"

default_args = {
    "start_date": datetime(2024, 1, 2),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

PSI_THRESHOLD = 0.20

def infer_pipeline(snapshot: str) -> str:
    cmd = [
        "docker", "exec", CONTAINER,
        "python", "pipelines/run_inference_and_monitor_pipeline.py",
        "--snapshotdate",  snapshot,
        "--baselinepath",  BASELINE,
        "--combinedpath",  "datamart/gold/combined_labeled.parquet",
        "--modeldir",      PROD_DIR,
        "--quiet",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = out.strip().split("\n")[-1]
    logging.info("Inference payload: %s", payload)
    return payload

def drift_decision(**ctx):
    ti  = ctx["ti"]
    raw = ti.xcom_pull(task_ids="run_infer_monitor") or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logging.error("Bad JSON from inference; forcing retrain")
        ti.xcom_push(key="drift_info", value={"error": "bad_json"})
        return "trigger_retrain"

    psi       = data.get("psi_max", 1.0)
    psi_ratio = psi / PSI_THRESHOLD

    if psi_ratio < 0.5:
        band, branch = "safe", "continue_pipeline"
    elif psi_ratio < 1.0:
        band, branch = "warning", "warning_alert"
    else:
        band, branch = "critical","trigger_retrain"

    drift_info = {
        "psi":           psi,
        "psi_threshold": PSI_THRESHOLD,
        "psi_ratio":     round(psi_ratio, 3),
        "band":          band,
        "branch_to":     branch
    }
    logging.info(
        "Drift Decision → band=%s; PSI=%.3f/%.3f(%.3fx)",
        band, psi, PSI_THRESHOLD, psi_ratio
    )
    ti.xcom_push(key="drift_info", value=drift_info)
    return branch

def _noop():
    logging.info("Safe band: pipeline continues without alert.")

with DAG(
    dag_id="inference_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 6, 1),
    catchup=True,
    schedule="@monthly",
    tags=["ml", "training", "inference"],
    description="Monthly inference + drift detection + reporting",
) as dag:

    run_infer_monitor = PythonOperator(
        task_id="run_infer_monitor",
        python_callable=infer_pipeline,
        op_args=["{{ dag_run.conf.get('snapshot', '2024-01-01') }}"],
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
            f"docker exec {CONTAINER} python pipelines/export_run_metrics.py "
            f"--experiment inference-monitoring "
            f"--outdir {REPORT_DIR}{{{{ ds }}}}"
        ),
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
 
    build_report = BashOperator(
        task_id="build_report",
        bash_command=(
            f"docker exec {CONTAINER} python pipelines/generate_performance_report.py "
            f"--metrics {REPORT_DIR}{{{{ ds }}}}/metrics.parquet "
            f"--outdir {REPORT_DIR}{{{{ ds }}}}"
        ),
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # to-do: replace with email
    warning_alert = EmptyOperator(
        task_id="warning_alert",
        trigger_rule="none_failed",
        do_xcom_push=False,
        on_execute_callback=lambda context: (_ for _ in ()).throw(
            Exception(
                f"Model drift warning on {context['ds']}: "
                f"PSI ratio in warning band. Please review report."
            )
        )
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="train_ml_pipeline",
        conf={"snapshot": "{{ ds }}"},
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # backfill_inference = BashOperator(
    #     task_id="backfill_inference",
    #     bash_command=(
    #         f"docker exec {CONTAINER} python "
    #         "pipelines/run_inference_and_monitor_pipeline.py "
    #         f"--baselinepath {BASELINE} "
    #         f"--combinedpath datamart/gold/combined_labeled.parquet "
    #         f"--modeldir {PROD_DIR} "
    #         "--quiet"
    #     ),
    #     trigger_rule=TriggerRule.ALL_SUCCESS,
    # )

    # backfill_inference >> drift_branch
    run_infer_monitor >> drift_branch
    drift_branch >> continue_pipeline >> export_metrics >> build_report
    drift_branch >> warning_alert >> export_metrics
    drift_branch >> trigger_retrain