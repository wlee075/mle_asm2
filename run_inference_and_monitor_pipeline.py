import pandas as pd
import numpy as np
import json
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import mlflow
import os
import sys
import pickle
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
logging.basicConfig(level=logging.INFO)

def create_spark_session():
    spark = SparkSession.builder \
        .appName("InferMonitor") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def generate_first_of_month_dates(start: str, end: str) -> list:
    sd = datetime.strptime(start, "%Y-%m-%d").replace(day=1)
    ed = datetime.strptime(end,   "%Y-%m-%d").replace(day=1)
    dates = []
    while sd <= ed:
        dates.append(sd.strftime("%Y-%m-%d"))
        sd += relativedelta(months=1)
    return dates


def find_model_artifact(model_dir: str, snapshot: str) -> str:
    suffix = f"_{snapshot}.pkl"
    for f in os.listdir(model_dir):
        if f.endswith(suffix):
            return os.path.join(model_dir, f)
    raise FileNotFoundError(f"No artifact for {snapshot}")

def compute_psi(baseline: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    breakpoints = np.linspace(0, 100, buckets + 1)
    bin_edges = np.percentile(baseline, breakpoints)
    base_pct = np.histogram(baseline, bins=bin_edges)[0] / len(baseline)
    curr_pct = np.histogram(current,  bins=bin_edges)[0] / len(current)
    base_pct = np.where(base_pct == 0, 1e-8, base_pct)
    curr_pct = np.where(curr_pct == 0, 1e-8, curr_pct)
    psi_vals = (base_pct - curr_pct) * np.log(base_pct / curr_pct)
    return float(np.sum(psi_vals))

def load_baseline(baseline_path: str, feature_cols: list) -> pd.DataFrame:
    # load baseline features used for psi computation
    ext = os.path.splitext(baseline_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(baseline_path)
    elif ext in (".parq", ".parquet"):
        spark = SparkSession.builder.getOrCreate()
        df = spark.read.parquet(baseline_path).toPandas()
    else:
        raise ValueError(f"Unsupported baseline extension: {ext}")
    return df[feature_cols]


def run_for_snapshot(
        spark: SparkSession,
        snapshot: str,
        baselinepath: str,
        featurepath: str,
        model_dir: str,
        psi_threshold: float = 0.2
):
    logging.info("=== Processing snapshot %s ===", snapshot)
    try:
        artifact_path = find_model_artifact(model_dir, snapshot)
    except FileNotFoundError as e:
        logging.error(str(e))
        return

    with open(artifact_path, "rb") as f:
        artefact = pickle.load(f)

    feats_sdf = (
        spark.read.csv(
            featurepath, 
            header=True, 
            inferSchema=True).filter(col("snapshot_date") == datetime.strptime(snapshot, "%Y-%m-%d"))
    )
    row_cnt = feats_sdf.count()
    if row_cnt == 0:
        logging.warning("No feature rows for %s – skipping", snapshot)
        return 0.0
    logging.info("Loaded %s feature rows for %s", row_cnt, snapshot)

    feats_pdf = feats_sdf.toPandas()
    feature_cols = [c for c in feats_pdf.columns if c.startswith("fe_")]
    baseline_pdf = load_baseline(baselinepath, feature_cols)
    psi_vals = {}
    psi_max = 0.0
    for col_name in feature_cols:
        base_vals = baseline_pdf[col_name].dropna().values
        curr_vals = feats_pdf[col_name].dropna().values
        if base_vals.size and curr_vals.size:
            psi = compute_psi(base_vals, curr_vals)
            mlflow.log_metric(f"psi_{col_name}", psi)
            psi_vals[col_name] = psi
            psi_max = max(psi_max, psi)
            
    if psi_vals:
        psi_mean = float(np.mean(list(psi_vals.values())))
        psi_max  = max(psi_vals.values())
        mlflow.log_metric("psi_mean", psi_mean)
        mlflow.log_metric("psi_max",  psi_max)

        if psi_max >= psi_threshold:
            logging.warning(
                "PSI drift threshold hit (max=%.3f, mean=%.3f). Abandoning "
                "inference for snapshot %s", psi_max, psi_mean, snapshot
            )
    
    scaler = artefact["preprocessing_transformers"]["stdscaler"]
    X_inf = scaler.transform(feats_pdf[feature_cols])

    model = artefact["model"]
    preds = model.predict_proba(X_inf)[:, 1]

    mlflow.log_metric("pred_mean", float(preds.mean()))
    mlflow.log_metric("pred_std",  float(preds.std()))
    out_pdf = feats_pdf[["Customer_ID", "snapshot_date"]].copy()
    out_pdf["model_name"] = os.path.basename(artifact_path)
    out_pdf["prediction"] = preds

    gold_dir = os.path.join(
        "datamart/gold/model_predictions",
        os.path.basename(artifact_path).replace(".pkl", ""),
    )
    os.makedirs(gold_dir, exist_ok=True)

    out_sdf = spark.createDataFrame(out_pdf)
    out_path = os.path.join(gold_dir, f"predictions_{snapshot}.parquet")
    out_sdf.write.mode("overwrite").parquet(out_path)

    logging.info("Wrote %s predictions rows to %s", len(out_pdf), out_path)
    return float(psi_max or 0.0)


def main(snapshotdate, baselinepath, featurepath, modeldir, quiet=False):
    if snapshotdate:
        snaps = [snapshotdate]
    else:
        today = datetime.now()
        end = today.strftime("%Y-%m-%d")
        start = (today - relativedelta(years=1)).strftime("%Y-%m-%d")
        snaps = generate_first_of_month_dates(start, end)

    global_max = 0.0
    spark = create_spark_session()
    for snap in snaps:
        psi_max_snapshot = run_for_snapshot(
            spark, snap, baselinepath, featurepath, modeldir
        )
        global_max = max(global_max, psi_max_snapshot)
        
    spark.stop()
    logging.info("--- Inference complete ---")
    if quiet:
        # last_active_run() can be None if MLflow logging was skipped
        run = mlflow.last_active_run()
        run_id = run.info.run_id if run else ""
        print(
            json.dumps(
                {"run_id": run_id,
                 "psi_max": round(global_max, 6)}
            )
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference + monitoring")
    parser.add_argument("--snapshotdate", required=False,
                        help="YYYY-MM-DD for single run; omit to backfill 1yr")
    parser.add_argument("--baselinepath", required=True)
    parser.add_argument("--featurepath",  required=True)
    parser.add_argument("--modeldir",     required=True)
    parser.add_argument("--quiet", action="store_true", help="Print final JSON payload for Airflow XCom")
    args = parser.parse_args()
    main(
        args.snapshotdate, 
        args.baselinepath, 
        args.featurepath,
        args.modeldir, 
        args.quiet
    )