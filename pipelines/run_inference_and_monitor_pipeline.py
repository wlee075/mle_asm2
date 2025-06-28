import pandas as pd
import numpy as np
import json
import argparse
from pyspark.sql import SparkSession
import mlflow
import os
import pickle
import logging
from datetime import datetime
from pyspark.sql.functions import col, lit
from dateutil.relativedelta import relativedelta
from sklearn.metrics import f1_score, accuracy_score

logging.basicConfig(level=logging.INFO)

CONTAINER = "data_ml_container_asm2"
DEFAULT_BASELINE = "data/baseline_features.csv"
DEFAULT_MODEL_DIR = "model_bank/production/"
REPORT_DIR = "reports/"
PSI_THRESHOLD = 0.20


def create_spark_session():
    spark = SparkSession.builder \
        .appName("InferMonitor") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def generate_first_of_month_dates(start: str, end: str) -> list:
    sd = datetime.strptime(start, "%Y-%m-%d").replace(day=1)
    ed = datetime.strptime(end, "%Y-%m-%d").replace(day=1)
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
    curr_pct = np.histogram(current, bins=bin_edges)[0] / len(current)
    base_pct = np.where(base_pct == 0, 1e-8, base_pct)
    curr_pct = np.where(curr_pct == 0, 1e-8, curr_pct)
    psi_vals = (base_pct - curr_pct) * np.log(base_pct / curr_pct)
    return float(np.sum(psi_vals))


def load_baseline(baseline_path: str, feature_cols: list) -> pd.DataFrame:
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
    combined_path: str,
    model_dir: str,
    reportdir: str,
    psi_threshold: float = PSI_THRESHOLD
) -> float:
    snap_dt = datetime.strptime(snapshot, "%Y-%m-%d")
    logging.info("=== Processing snapshot %s ===", snapshot)

    artifact_path = find_model_artifact(model_dir, snapshot)
    with open(artifact_path, "rb") as f:
        artefact = pickle.load(f)

    combined_sdf = (
        spark.read.parquet(combined_path)
            .filter(col("feature_snapshot_date") == lit(snap_dt))
    )
    pdf = combined_sdf.toPandas()
    if pdf.empty:
        logging.warning("No data found for %s in combined table", snapshot)
        return 0.0

    feature_cols = [c for c in pdf.columns if c.startswith("fe_")]
    X_pdf = pdf[feature_cols]
    y_true = pdf["label"]
    logging.info("Scoring on %d rows for snapshot %s", len(pdf), snapshot)

    baseline_pdf = load_baseline(baselinepath, feature_cols)
    psi_vals = {}
    for col_name in feature_cols:
        psi = compute_psi(
            baseline_pdf[col_name].dropna().values,
            X_pdf[col_name].dropna().values
        )
        mlflow.log_metric(f"psi_{col_name}", psi)
        psi_vals[col_name] = psi

    psi_max = max(psi_vals.values())
    mlflow.log_metric("psi_mean", float(np.mean(list(psi_vals.values()))))
    mlflow.log_metric("psi_max", float(psi_max))

    if psi_max >= psi_threshold:
        logging.warning(
            "PSI drift threshold hit (psi_max=%.3f ≥ threshold=%.3f). "
            "Aborting inference for %s", psi_max, psi_threshold, snapshot
        )
        return psi_max 

    scaler = artefact["preprocessing_transformers"]["stdscaler"]
    X_inf = scaler.transform(X_pdf)
    preds = artefact["model"].predict_proba(X_inf)[:, 1]
    mlflow.log_metric("pred_mean", float(preds.mean()))
    mlflow.log_metric("pred_std", float(preds.std()))

    y_pred = (preds >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    mlflow.log_metric("f1_score", float(f1))
    mlflow.log_metric("accuracy", float(acc))

    out = {"snapshot": snapshot, "psi_max": float(psi_max), "f1_score": float(f1), "accuracy": float(acc)}
    out_dir = os.path.join(reportdir, snapshot)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "performance_metrics.json"), "w") as f:
        json.dump(out, f)

    out["model_type"] = artefact["model"].__class__.__name__.lower()
    metrics_path = os.path.join(reportdir, "model_monitoring_metrics.parquet")
    pdf = pd.DataFrame([out])
    if os.path.exists(metrics_path):
        old = pd.read_parquet(metrics_path)
        pdf = pd.concat([old, pdf], ignore_index=True)
    pdf.to_parquet(metrics_path, index=False)
    return float(psi_max)


def main(
    snapshotdate,
    baselinepath,
    combinedpath,
    modeldir,
    reportdir,
    psi_threshold=PSI_THRESHOLD,
    quiet=False
):
    mlflow.set_experiment("inference-monitoring")
    mlflow.start_run(run_name=f"infer_{snapshotdate or 'backfill'}")
    if snapshotdate:
        snaps = [snapshotdate]
    else:
        today = datetime.now()
        end = today.strftime("%Y-%m-%d")
        start = (today - relativedelta(years=1)).strftime("%Y-%m-%d")
        snaps = generate_first_of_month_dates(start, end)

    spark = create_spark_session()
    global_max = 0.0
    for snap in snaps:
        psi_val = run_for_snapshot(
            spark,
            snap,
            baselinepath,
            combinedpath,
            modeldir,
            reportdir,
            psi_threshold
        )
        global_max = max(global_max, psi_val)
    spark.stop()
    mlflow.end_run()

    if quiet:
        run = mlflow.last_active_run()
        rid = run.info.run_id if run else ""
        print(json.dumps({"run_id": rid, "psi_max": round(global_max,6)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference + monitoring + evaluation")
    parser.add_argument("--snapshotdate", help="YYYY-MM-DD for single run; omit to backfill 1yr")
    parser.add_argument("--baselinepath", default=DEFAULT_BASELINE)
    parser.add_argument("--combinedpath", required=True, help="Path to pre-joined features+labels parquet")
    parser.add_argument("--modeldir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--quiet", action="store_true", help="Print JSON for XCom")
    args = parser.parse_args()
    main(
        args.snapshotdate,
        args.baselinepath,
        args.combinedpath,
        args.modeldir,
        REPORT_DIR,
        PSI_THRESHOLD,
        args.quiet
    )