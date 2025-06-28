import os
from pyspark.sql import SparkSession
from utils.silver_processing import (
    clean_financials_table,
    clean_attributes_table,
    clean_clickstream_table,
    clean_loans_table
)

def create_spark_session():
    spark = SparkSession.builder\
        .appName("LoanFeaturePipeline")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

if __name__ == "__main__":
    spark = create_spark_session()
    clean_financials_table(spark)
    clean_attributes_table(spark)
    clean_clickstream_table(spark)
    clean_loans_table(spark)