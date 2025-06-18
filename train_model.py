#!/usr/bin/env python3
"""
Train either Logistic-Regression or XGBoost for a *single* snapshot.
If the requested snapshot has no feature rows (or no labels), the code
walks backward month-by-month until it finds the most-recent snapshot
with data – then trains on that instead.
"""
import argparse
import glob
import logging
import os
import pickle
from datetime import datetime

import mlflow
import pyspark
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    filename="train_model.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder.appName("ModelTraining")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def latest_snapshot_with_rows(
    spark: SparkSession, feature_csv: str, target_snap: str
) -> str | None:
    """
    Walk backwards month-by-month (starting with *target_snap*) until
    we find a snapshot that has ≥1 feature rows.  Return its date in
    YYYY-MM-DD or None if nothing exists.
    """
    snap_dt = datetime.strptime(target_snap, "%Y-%m-%d")
    while snap_dt.year >= 2000:  # safety guard
        cnt = (
            spark.read.csv(feature_csv, header=True, inferSchema=True)
            .filter(col("snapshot_date") == snap_dt)
            .limit(1)
            .count()
        )
        if cnt:
            return snap_dt.strftime("%Y-%m-%d")
        snap_dt -= relativedelta(months=1)
    return None

def main(featurepath: str,
         labeldir: str,
         model_dir: str,
         modeltype: str,
         snapshotdate: str):

    spark = create_spark_session()
    logging.info("Requested snapshot: %s", snapshotdate)

    valid_snap = snapshotdate
    feat_cnt = (
        spark.read.csv(featurepath, header=True, inferSchema=True)
        .filter(col("snapshot_date") == datetime.strptime(snapshotdate, "%Y-%m-%d"))
        .limit(1)
        .count()
    )

    if feat_cnt == 0:
        logging.warning("No feature rows for %s. searching for fallback", snapshotdate)
        valid_snap = latest_snapshot_with_rows(spark, featurepath, snapshotdate)
        if valid_snap is None:
            logging.error("Could not find any snapshot with features – aborting")
            return

    feat_sdf = (
        spark.read.csv(featurepath, header=True, inferSchema=True)
        .filter(col("snapshot_date") == datetime.strptime(valid_snap, "%Y-%m-%d"))
    )
    feat_pdf = feat_sdf.toPandas()
    feature_cols = [c for c in feat_pdf.columns if c.startswith("fe_")]

    label_files = glob.glob(os.path.join(labeldir, "*.parquet"))
    if not label_files:
        logging.error("No label parquet files in %s. aborting", labeldir)
        return
    label_pdf = (
        spark.read.parquet(*label_files)
        .filter(col("snapshot_date") == datetime.strptime(valid_snap, "%Y-%m-%d"))
        .toPandas()[["Customer_ID", "label"]]
    )

    data = feat_pdf.merge(label_pdf, on="Customer_ID", how="inner")
    if data.empty:
        logging.error("No matching features/labels for snapshot %s. aborting", valid_snap)
        return

    X, y = data[feature_cols].values, data["label"].values
    stratify = y if len(set(y)) > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
    mlflow.set_tracking_uri("file:///ml_bank")
    mlflow.set_experiment("model-training-full")
    with mlflow.start_run(run_name=f"train_{modeltype}_{valid_snap}"):

        mlflow.set_tag("snapshot_date", valid_snap)
        mlflow.set_tag("model_type", modeltype)

        if modeltype == "logistic_regression":
            model = LogisticRegression(max_iter=1000).fit(X_tr_s, y_tr)
        else:
            model = xgb.XGBClassifier(use_label_encoder=False,
                                      eval_metric="logloss").fit(X_tr_s, y_tr)

        auc = roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1])
        mlflow.log_metric("auc", auc)

        os.makedirs(model_dir, exist_ok=True)
        artefact_path = os.path.join(model_dir, f"{modeltype}_{valid_snap}.pkl")
        with open(artefact_path, "wb") as fp:
            pickle.dump(
                {"preprocessing_transformers": {"stdscaler": scaler}, "model": model},
                fp,
            )

        mlflow.log_artifact(artefact_path, artifact_path="model_store")
        logging.info("Model %s saved to %s (AUC = %.4f)", modeltype, artefact_path, auc)

    spark.stop()
    logging.info("Training finished for snapshot %s (actual data snapshot %s)",
                 snapshotdate, valid_snap)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train one model for a given snapshot")
    p.add_argument("--featurepath", required=True)
    p.add_argument("--labeldir",    required=True)
    p.add_argument("--modeldir",    required=True)
    p.add_argument("--modeltype",   required=True,
                   choices=["logistic_regression", "xgboost"])
    p.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD")
    args = p.parse_args()
    main(args.featurepath, args.labeldir, args.modeldir,
         args.modeltype, args.snapshotdate)
