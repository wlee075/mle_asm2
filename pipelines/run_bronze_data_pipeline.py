import os
from pyspark.sql import SparkSession
from utils.bronze_processing import ingest_bronze_tables

def create_spark_session():
    spark = SparkSession.builder\
        .appName("LoanFeaturePipeline")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def init_datamart():
    for layer in ["datamart/bronze", "datamart/silver", "datamart/gold"]:
        os.makedirs(layer, exist_ok=True)

if __name__ == "__main__":
    spark = create_spark_session()
    init_datamart()

    # Bronze
    ingest_bronze_tables(spark)
    print("Bronze complete!")