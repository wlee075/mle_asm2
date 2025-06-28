import logging
import os
import argparse
from pyspark.sql.functions import col, lit
from pyspark.sql import SparkSession
def create_spark_session():
    spark = SparkSession.builder\
        .appName("LoanFeaturePipeline")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def write_baseline_features(gold_feats, train_date, baseline_path):
    baseline_df = gold_feats.filter(col("feature_snapshot_date") == train_date)
    feature_cols = [c for c in baseline_df.columns if c.startswith("fe_")]
    baseline_pd = baseline_df.select(feature_cols).toPandas()
    logging.info("Baseline rows: %d", baseline_pd.shape[0])
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    baseline_pd.to_csv(baseline_path, index=False)
    logging.info("Wrote baseline to %s (%d rows)", baseline_path, baseline_pd.shape[0])
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    gold_feats.select(feature_cols).toPandas().to_csv(baseline_path, index=False)
    logging.info(f"Wrote full baseline to {baseline_path}")

    
if __name__ == "__main__":
    spark = create_spark_session()
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselinepath", required=True)
    args = parser.parse_args()
    baseline_path = args.baselinepath
    try:
        dates = (spark.read.parquet("data/gold/combined_labeled.parquet")
                 .select("feature_snapshot_date")
                 .distinct()
                 .orderBy("feature_snapshot_date")
                 .collect())
    except Exception as e:
        logging.exception(f"Unexpected error getting collection of feature snapshot dates: {e}")
    try:
        # get most recent date
        train_date = dates[-1].feature_snapshot_date
    except Exception as e:
        logging.exception(f"Unexpected error get latest baseline date: {e}")
    try:
        gold_feats = spark.read.parquet("data/gold/combined_labeled.parquet")
    except Exception as e:
        logging.exception(f"Unexpected error getting gold features: {e}")
    try:
        write_baseline_features(gold_feats, train_date, baseline_path)
    except Exception as e:
         logging.exception(f"Unexpected error writing gold features: {e}")


         