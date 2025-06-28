from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta

CONTAINER = "data_ml_container_asm2"
FEATURES  = "data/feature_clickstream.csv"
LABEL_DIR = "datamart/gold/label_store/"
CAND_DIR  = "model_bank/"
PROD_DIR  = "model_bank/production/"

default_args = {
    "start_date": datetime(2024, 1, 2),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="train_ml_pipeline",
    default_args=default_args,
    schedule="@once",
    catchup=True,
    tags=["ml", "retrain"],
    description="Re-train candidate models and push best into production",
) as dag:

    train_lr = BashOperator(
        task_id="train_logistic_regression",
        bash_command=(
            "docker exec {{ params.container }} python pipelines/train_model.py "
            "--modeltype logistic_regression "
            "--snapshotdate {{ dag_run.conf.get('snapshot', '2024-01-01') }} "
            "--featurepath {{ params.features }} "
            "--labeldir {{ params.label_dir }} "
            "--modeldir {{ params.cand_dir }}"
        ),
       params={
           "container": CONTAINER,
           "features": FEATURES,
           "label_dir": LABEL_DIR,
           "cand_dir": CAND_DIR,
       }
    )

    train_xgb = BashOperator(
        task_id="train_xgboost",
        bash_command=(
            "docker exec {{ params.container }} python pipelines/train_model.py "
            "--modeltype xgboost "
            "--snapshotdate {{ dag_run.conf.get('snapshot', '2024-01-01') }} "
            "--featurepath {{ params.features }} "
            "--labeldir {{ params.label_dir }} "
            "--modeldir {{ params.cand_dir }}"
        ),
       params={
           "container": CONTAINER,
           "features": FEATURES,
           "label_dir": LABEL_DIR,
           "cand_dir": CAND_DIR,
       }
    )

    select_best = BashOperator(
        task_id="select_best_model",
        bash_command=(
            f"docker exec {CONTAINER} python pipelines/select_model.py "
            f"--candidatedir {CAND_DIR} --productiondir {PROD_DIR} "
            "--startdate {{ dag_run.conf.get('snapshot', macros.ds_format(ds, '%Y-%m-%d','%Y-%m-01')) }} "
            "--enddate   {{ dag_run.conf.get('snapshot', macros.ds_format(ds, '%Y-%m-%d','%Y-%m-01')) }} "
        ),
    )

    trigger_inference = TriggerDagRunOperator(
        task_id="trigger_inference",
        trigger_dag_id="inference_pipeline",
        conf={"snapshot": "{{ dag_run.conf.get('snapshot', macros.ds_format(ds, '%Y-%m-%d','%Y-%m-01')) }}"},
        trigger_rule="all_success",
    )
    

    [train_lr, train_xgb] >> select_best >> trigger_inference