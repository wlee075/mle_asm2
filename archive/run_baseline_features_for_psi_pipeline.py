import logging
from pyspark.sql import SparkSession
def create_spark_session():
    spark = SparkSession.builder\
        .appName("LoanFeaturePipeline")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

if __name__ == "__main__":
    spark = create_spark_session()
    try:
        dates = (spark.read.parquet("datamart/gold/feature_store").select("feature_snapshot_date").distinct().orderBy("feature_snapshot_date").collect())
    except Exception as e:
        logging.exception(f"Unexpected error getting collection of feature snapshot dates: {e}")
    # get most recent date
    try:
        train_date = dates[-1].feature_snapshot_date
    except Exception as e:
        logging.exception(f"Unexpected error get latest baseline date: {e}")
    try:
        gold_feats = spark.read.parquet("datamart/gold/feature_store")
        gold_feats.select("feature_snapshot_date") \
                  .distinct() \
                  .orderBy("feature_snapshot_date") \
                  .show(truncate=False)
    except Exception as e:
        logging.exception(f"Unexpected error getting gold features: {e}")
    baseline_df = gold_feats.filter(col("feature_snapshot_date") == train_date)
    feature_cols = [c for c in baseline_df.columns if c.startswith("fe_")]
    baseline_pd = baseline_df.select(feature_cols).toPandas()
    logging.info("Baseline rows:", baseline_pd.shape[0])
    baseline_pd.to_csv("data/baseline_features.csv", index=False)
    logging.info("Wrote baseline_features.csv with", baseline_pd.shape[0], "rows.")
    baseline_cols = [c for c in gold_feats.columns if c.startswith("fe_")]
    baseline_pd = gold_feats.select(baseline_cols).toPandas()
    os.makedirs("data", exist_ok=True)
    baseline_path = "data/baseline_features.csv"
    baseline_pd.to_csv(baseline_path, index=False)
    logging.info(f"Wrote baseline to {baseline_path}")
    