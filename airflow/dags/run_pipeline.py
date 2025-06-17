from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="run_data_and_ml_pipeline",
    description="Run Data and ML Pipeline",
    start_date=datetime(2025, 6, 1),
    catchup=False,
) as dag:

    run_bronze_data_pipeline = BashOperator(
        task_id="run_bronze_data_pipeline",
        # data_ml_container is defined in the docker-compose.yml file
        bash_command="docker exec data_ml_container_asm2 python run_bronze_data_pipeline.py",
    )

    run_silver_data_pipeline = BashOperator(
        task_id="run_silver_data_pipeline",
        # data_ml_container is defined in the docker-compose.yml file
        bash_command="docker exec data_ml_container_asm2 python run_silver_data_pipeline.py",
    )

    run_gold_data_pipeline = BashOperator(
        task_id="run_gold_data_pipeline",
        # data_ml_container is defined in the docker-compose.yml file
        bash_command="docker exec data_ml_container_asm2 python run_gold_data_pipeline.py",
    )

    # run_online_feature_data_pipeline = BashOperator(
    #     task_id="run_online_feature_data_pipeline",
    #     # data_ml_container is defined in the docker-compose.yml file
    #     bash_command="docker exec data_ml_container python run_online_feature_data_pipeline.py",
    # )

    # run_ml_training_pipeline = BashOperator(
    #     task_id="run_ml_training_pipeline",
    #     bash_command="docker exec data_ml_container python run_ml_pipeline.py",
    # )

    (
        run_bronze_data_pipeline
        >> run_silver_data_pipeline
        >> run_gold_data_pipeline
        # >> run_online_feature_data_pipeline
        # >> run_ml_training_pipeline
    )