import os
from pyspark.sql import SparkSession
from utils.gold_processing import (
    build_label_store,
    build_feature_store
)
from pyspark.sql.functions import concat_ws, date_format, col
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
    logging.info("Gold complete!")
    features = spark.read.parquet("datamart/gold/feature_store")
    labels = spark.read.parquet("datamart/gold/label_store")
    features_with_id = features.withColumn(
        "loan_id",
        concat_ws("_",
            col("Customer_ID"),
            date_format(col("feature_snapshot_date"), "yyyy_MM_dd")
        )
    )
    combined = (
        features_with_id.join(labels.select("loan_id", "label"), on="loan_id", how="inner")
    )
    combined.write.mode("overwrite") \
            .parquet("datamart/gold/combined_labeled.parquet")
    logging.info("combined_labeled.parquet")