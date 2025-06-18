import argparse
import os
import pickle
import glob
import mlflow
import pyspark
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import logging

# Basic logging
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


def main(featurepath: str, labeldir: str, model_dir: str, modeltype: str, snapshotdate: str):
    logging.info(f"--- Training {modeltype} for snapshot {snapshotdate} ---")
    spark = create_spark_session()

    # Load features for given snapshot
    feat_sdf = (
        spark.read.csv(featurepath, header=True, inferSchema=True)
             .filter(col("snapshot_date") == datetime.strptime(snapshotdate, "%Y-%m-%d"))
    )
    feat_pdf = feat_sdf.toPandas()
    feature_cols = [c for c in feat_pdf.columns if c.startswith('fe_')]

    # Load label store and filter for this snapshot
    files = glob.glob(os.path.join(labeldir, "*.parquet"))
    if not files:
        logging.error(f"No label files in {labeldir}; aborting.")
        return
    label_pdf = spark.read.option("header","true").parquet(*files).toPandas()
    label_pdf = label_pdf[label_pdf["snapshot_date"] == snapshotdate][["Customer_ID","label"]]

    # Join features + labels
    data = feat_pdf.merge(label_pdf, on=["Customer_ID"], how="inner")
    if data.empty:
        logging.error(f"No data for snapshot {snapshotdate}; aborting.")
        return
    X, y = data[feature_cols].values, data['label'].values

    # Train/test split
    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    # MLflow logging
    mlflow.set_experiment("model-training-full")
    with mlflow.start_run(run_name=f"train_{modeltype}_{snapshotdate}"):
        mlflow.set_tag("snapshot_date", snapshotdate)
        mlflow.set_tag("model_type", modeltype)

        # Train model
        if modeltype == 'logistic_regression':
            model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
        else:
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train_s, y_train)

        # Evaluate
        auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:,1])
        mlflow.log_metric("auc", auc)

        # Save snapshot artifact
        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, f"{modeltype}_{snapshotdate}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump({'preprocessing_transformers':{'stdscaler':scaler}, 'model':model}, f)
        mlflow.log_artifact(out_path, artifact_path="model_store")
        logging.info(f"Saved {modeltype} model to {out_path} with AUC {auc}")

    spark.stop()
    logging.info(f"--- Completed training for {modeltype} snapshot {snapshotdate} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train one model for a given snapshot")
    parser.add_argument("--featurepath", required=True)
    parser.add_argument("--labeldir",    required=True)
    parser.add_argument("--modeldir",    required=True)
    parser.add_argument("--modeltype",   required=True, choices=['logistic_regression','xgboost'])
    parser.add_argument("--snapshotdate",required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(
        args.featurepath,
        args.labeldir,
        args.modeldir,
        args.modeltype,
        args.snapshotdate
    )