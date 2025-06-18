import argparse
import os
import pandas as pd
import pickle
import numpy as np
import pprint
import mlflow
import mlflow.xgboost
import pyspark
from pyspark.sql.functions import col
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging

# to call this script:
# python model_train.py \
#   --snapshotdate 2024-09-01 \
#   --modelname fraud_xgb.pkl \
#   --baselinepath data/baseline_features.csv

def compute_psi(baseline: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    breakpoints = np.linspace(0, 100, buckets + 1)
    bin_edges = np.percentile(baseline, breakpoints)
    base_counts = np.histogram(baseline, bins=bin_edges)[0] / len(baseline)
    curr_counts = np.histogram(current,  bins=bin_edges)[0] / len(current)
    base_counts = np.where(base_counts == 0, 1e-8, base_counts)
    curr_counts = np.where(curr_counts == 0, 1e-8, curr_counts)
    psi_vals = (base_counts - curr_counts) * np.log(base_counts / curr_counts)
    return float(np.sum(psi_vals))


def load_baseline(baseline_path: str, feature_cols: list):
    ext = os.path.splitext(baseline_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(baseline_path)
    elif ext in ('.parq', '.parquet'):
        spark = pyspark.sql.SparkSession.builder.getOrCreate()
        df = spark.read.parquet(baseline_path).toPandas()
    else:
        raise ValueError(f"Unsupported baseline file extension: {ext}")
    return df[feature_cols]


def main(snapshotdate, modelname, baselinepath):
    print('\n\n--- Starting job ---\n\n')
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    config = {
        "snapshot_date_str": snapshotdate,
        "snapshot_date": datetime.strptime(snapshotdate, "%Y-%m-%d"),
        "model_name": modelname,
        "model_bank_directory": "model_bank/"
    }
    config["model_artefact_filepath"] = os.path.join(
        config["model_bank_directory"], config["model_name"]
    )
    pprint.pprint(config)

    mlflow.set_experiment("fraud-inference")
    with mlflow.start_run(run_name=f"infer_{modelname}_{snapshotdate}"):
        mlflow.log_param("snapshot_date", config["snapshot_date_str"])
        mlflow.log_param("model_name",    config["model_name"])

        # Load model
        with open(config["model_artefact_filepath"], 'rb') as f:
            model_artefact = pickle.load(f)
        mlflow.log_artifact(config["model_artefact_filepath"], artifact_path="model_artifact")

        # Load features
        feature_location = "data/feature_clickstream.csv"
        features_sdf = (
            spark.read.csv(feature_location, header=True, inferSchema=True)
                 .filter(col("snapshot_date") == config["snapshot_date"] )
        )
        count_rows = features_sdf.count()
        print(f"Extracted {count_rows} rows for {snapshotdate}")
        mlflow.log_metric("num_rows_loaded", count_rows)

        features_pdf = features_sdf.toPandas()
        feature_cols = [c for c in features_pdf.columns if c.startswith('fe_')]

        # Baseline
        baseline_pdf = load_baseline(baselinepath, feature_cols)

        # Compute PSI per feature
        psi_vals = {}
        for feat in feature_cols:
            base_vals = baseline_pdf[feat].dropna().values
            curr_vals = features_pdf[feat].dropna().values
        
            if len(base_vals) == 0 or len(curr_vals) == 0:
                print(f"[WARNING] skipping PSI for {feat}: "
                      f"{len(base_vals)} baseline rows, {len(curr_vals)} current rows")
                continue
        
            psi = compute_psi(baseline=base_vals, current=curr_vals, buckets=10)
            mlflow.log_metric(f"psi_{feat}", psi)
            psi_vals[feat] = psi

        # Save PSI report
        psi_df = pd.DataFrame({"feature": list(psi_vals.keys()), "psi": list(psi_vals.values())})
        psi_csv = f"/tmp/psi_report_{snapshotdate}.csv"
        logging.info(f'saved psi to {psi_csv}')
        psi_df.to_csv(psi_csv, index=False)
        mlflow.log_artifact(psi_csv, artifact_path="drift_reports")

        # Transform & predict
        transformer = model_artefact["preprocessing_transformers"]["stdscaler"]
        X_inf = transformer.transform(features_pdf[feature_cols])
        mlflow.log_metric("num_rows_after_transform", X_inf.shape[0])

        model = model_artefact["model"]
        y_pred = model.predict_proba(X_inf)[:, 1]
        mlflow.log_metric("pred_mean", float(np.mean(y_pred)))
        mlflow.log_metric("pred_std",  float(np.std(y_pred)))

        # Save predictions
        out_pdf = features_pdf[["Customer_ID", "snapshot_date"]].copy()
        out_pdf["model_name"] = config["model_name"]
        out_pdf["model_predictions"] = y_pred
        preds_csv = f"/tmp/preds_{modelname}_{snapshotdate}.csv"
        
        out_pdf.to_csv(preds_csv, index=False)
        mlflow.log_artifact(preds_csv, artifact_path="predictions")

        # Write to gold
        gold_dir = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
        os.makedirs(gold_dir, exist_ok=True)
        part = f"{config['model_name'][:-4]}_predictions_{snapshotdate.replace('-','_')}.parquet"
        gold_path = os.path.join(gold_dir, part)
        spark.createDataFrame(out_pdf) \
             .write.mode("overwrite") \
             .parquet(gold_path)
        print(f"Saved to: {gold_path}")

    spark.stop()
    print('\n\n--- Job completed ---\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job with PSI drift monitoring")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model pickle file")
    parser.add_argument("--baselinepath", type=str, required=True, help="baseline features file (CSV or Parquet)")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname, args.baselinepath)
