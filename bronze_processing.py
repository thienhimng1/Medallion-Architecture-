from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
import os, re

def process_bronze(spark, folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("Không tìm thấy tệp CSV trong thư mục")

    dfs = []
    for file in csv_files:
        match = re.match(r'([A-Z]+)_(\d{1,2}-\d{1,2})', file)
        location, date = match.groups() if match else ('UNKNOWN', 'UNKNOWN')
        df = spark.read.option("header", "true").csv(os.path.join(folder_path, file))
        df = df.withColumn("price", lit(None)).withColumn("host_review_score", lit(None))
        df = df.drop("price", "host_review_score")
        df = df.withColumn('location', lit(location)).withColumn('date', lit(date))
        dfs.append(df)

    bronze_df = dfs[0]
    for df in dfs[1:]:
        bronze_df = bronze_df.unionByName(df)

    bronze_table_path = os.path.abspath("delta/bronze/hotels")
    os.makedirs(bronze_table_path, exist_ok=True)
    bronze_df.write.format("delta").mode("overwrite").save(bronze_table_path)
    spark.sql(f"CREATE TABLE IF NOT EXISTS bronze_hotels USING DELTA LOCATION '{bronze_table_path}'")
    return bronze_df