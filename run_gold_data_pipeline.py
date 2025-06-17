import os
from pyspark.sql import SparkSession
from utils.gold_processing import (
    build_label_store,
    build_feature_store
)
import logging


def create_spark_session():
    spark = SparkSession.builder\
        .appName("LoanFeaturePipeline")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

if __name__ == "__main__":
    spark = create_spark_session()
    build_label_store(spark, dpd_cutoff=30, mob_cutoff=6)
    build_feature_store(spark, dpd_cutoff=30, mob_cutoff=6)
    logger.info("Gold complete!")
    dates = (spark.read.parquet("datamart/gold/feature_store")
         .select("feature_snapshot_date")
         .distinct()
         .orderBy("feature_snapshot_date")
         .collect())
    # get most recent date
    train_date = dates[-1].feature_snapshot_date
    gold_feats = spark.read.parquet("datamart/gold/feature_store")
    gold_feats.select("feature_snapshot_date") \
              .distinct() \
              .orderBy("feature_snapshot_date") \
              .show(truncate=False)
    baseline_df = gold_feats.filter(col("feature_snapshot_date") == train_date)
    feature_cols = [c for c in baseline_df.columns if c.startswith("fe_")]
    baseline_pd = baseline_df.select(feature_cols).toPandas()
    logger.info("Baseline rows:", baseline_pd.shape[0])
    baseline_pd.to_csv("data/baseline_features.csv", index=False)
    logger.info("Wrote baseline_features.csv with", baseline_pd.shape[0], "rows.")