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
    logging.info("Gold complete!")