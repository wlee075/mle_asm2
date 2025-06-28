import argparse
import os
import pickle
import glob
import mlflow
import logging
import traceback
import shutil
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import roc_auc_score

INITIAL_TEST_DATE = "2024-02-01"
TRAIN_START_DATE = "2023-01-01"
TRAIN_END_DATE = "2023-12-01"
VAL_DATE = "2024-01-01"
BASELINE_DATE = "2024-01-01"

logging.basicConfig(
    filename="train_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def create_spark_session():
    spark = SparkSession.builder \
        .appName("ModelTraining") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def is_initial_run(snapshotdate: str) -> bool:
    return snapshotdate == INITIAL_TEST_DATE

def main(featurepath: str,
         labeldir:    str,
         model_dir:   str,
         modeltype:   str,
         snapshotdate:str):

    logging.info(f"--- Training {modeltype} for snapshot {snapshotdate} ---")
    spark = create_spark_session()

    feats_sdf = (
        spark.read
             .csv(featurepath, header=True, inferSchema=True)
             .withColumn("snapshot_date", col("snapshot_date").cast("date"))
    )

    files = glob.glob(os.path.join(labeldir, "*.parquet"))
    if not files:
        logging.error(f"No label files found in {labeldir}; aborting.")
        return

    label_sdf = (
        spark.read
             .parquet(*files)
             .withColumn("snapshot_date", col("snapshot_date").cast("date"))
             .select("Customer_ID", "snapshot_date", "label")
    )

    data_sdf = feats_sdf.join(
        label_sdf,
        on=["Customer_ID", "snapshot_date"],
        how="inner"
    )
    if data_sdf.rdd.isEmpty():
        logging.error(f"No merged rows up through {snapshotdate}; aborting.")
        spark.stop()
        return

    initial = is_initial_run(snapshotdate)

    if initial:
        train_sdf = data_sdf.filter(
            (col("snapshot_date") >= lit(TRAIN_START_DATE)) &
            (col("snapshot_date") <= lit(TRAIN_END_DATE))
        )
        val_sdf   = data_sdf.filter(col("snapshot_date") == lit(VAL_DATE))
        test_sdf  = data_sdf.filter(col("snapshot_date") == lit(INITIAL_TEST_DATE))

        train_pdf = train_sdf.toPandas()
        val_pdf   = val_sdf.toPandas()
        test_pdf  = test_sdf.toPandas()

        feature_cols = [c for c in train_pdf.columns if c.startswith("fe_")]

        X_train, y_train = train_pdf[feature_cols].values, train_pdf["label"].values
        X_val,   y_val   = val_pdf[feature_cols].values,   val_pdf["label"].values
        X_test,  y_test  = test_pdf[feature_cols].values,  test_pdf["label"].values

    else:
        data_pdf = data_sdf.toPandas()
        feature_cols = [c for c in data_pdf.columns if c.startswith("fe_")]

        X_full, y_full = data_pdf[feature_cols].values, data_pdf["label"].values
        stratify = y_full if len(set(y_full)) > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full,
            test_size=0.2,
            random_state=42,
            stratify=stratify
        )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    if initial:
        X_val_s = scaler.transform(X_val)

    mlflow.set_experiment("model-training-full")
    with mlflow.start_run(run_name=f"train_{modeltype}_{snapshotdate}"):
        mlflow.set_tag("snapshot_date", snapshotdate)
        mlflow.set_tag("model_type",    modeltype)
        mlflow.set_tag("initial_run",   str(initial))

        if modeltype == "logistic_regression":
            model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
        else:
            model = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss"
            ).fit(X_train_s, y_train)

        if initial:
            val_auc  = roc_auc_score(y_val, model.predict_proba(X_val_s)[:,1])
            test_auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:,1])
            mlflow.log_metric("val_auc",  round(val_auc,  4))
            mlflow.log_metric("test_auc", round(test_auc, 4))
            logging.info(f"Initial split → val_auc={val_auc:.4f}, test_auc={test_auc:.4f}")
        else:
            holdout_auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:,1])
            mlflow.log_metric("holdout_auc", round(holdout_auc, 4))
            logging.info(f"Retrain split → holdout_auc={holdout_auc:.4f}")

        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, f"{modeltype}_{snapshotdate}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({
                "preprocessing_transformers": {"stdscaler": scaler},"model": model}, f
            )
        mlflow.log_artifact(out_path, artifact_path="model_store")
        logging.info(f"Saved artifact to {out_path}")
    
        baseline_path = os.path.join(model_dir, f"{modeltype}_{BASELINE_DATE}.pkl")
        if not os.path.exists(baseline_path):
            shutil.copy(out_path, baseline_path)
            logging.info(f"Saved baseline model copy: {baseline_path}")
        prod_dir = os.path.join(model_dir, "production")
        os.makedirs(prod_dir, exist_ok=True)
        prod_baseline = os.path.join(prod_dir, os.path.basename(baseline_path))
        if not os.path.exists(prod_baseline):
            shutil.copy(baseline_path, prod_baseline)
            logging.info(f"Copied baseline into production: {prod_baseline}")

    spark.stop()
    logging.info(f"--- Completed training for {modeltype} snapshot {snapshotdate} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train one model for a given snapshot")
    parser.add_argument("--featurepath",   required=True)
    parser.add_argument("--labeldir",      required=True)
    parser.add_argument("--modeldir",      required=True)
    parser.add_argument("--modeltype",     required=True, choices=['logistic_regression','xgboost'])
    parser.add_argument("--snapshotdate",  required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    try:
        main(
            args.featurepath,
            args.labeldir,
            args.modeldir,
            args.modeltype,
            args.snapshotdate
        )
        
    except Exception:
        logging.exception("An error occurred during training")
        raise