import os, sys, requests, logging, argparse
from mlflow.tracking import MlflowClient

THRESHOLD = 0.20
AIRFLOW_API = os.getenv("AIRFLOW_API", "http://localhost:8080/api/v1/dags")

def latest_inference_run(exp_name="model-training-full"):
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    return runs[0] if runs else None

def trigger_dag(dag_id: str, conf: dict):
    url = f"{AIRFLOW_API}/{dag_id}/dagRuns"
    r = requests.post(url, json={"conf": conf})
    r.raise_for_status()
    logging.info("Triggered retrain DAG %s", dag_id)

def main(dag_to_trigger: str):
    run = latest_inference_run()
    if run is None:
        logging.warning("No inference run found")
        sys.exit(0)

    psi_max = run.data.metrics.get("psi_max", 0)
    if psi_max >= THRESHOLD:
        logging.warning("Drift %.3f ≥ threshold %.2f - triggering retrain", psi_max, THRESHOLD)
        trigger_dag(dag_to_trigger, {"drift": psi_max, "run_id": run.info.run_id})
    else:
        logging.info("PSI %.3f below threshold - all good", psi_max)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retrain-dag", default="run_data_and_ml_pipeline")
    main(p.parse_args().retrain_dag)