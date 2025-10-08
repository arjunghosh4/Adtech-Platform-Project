from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("KafkaToMinIOAllTopics") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

df_users = spark.read.parquet("s3a://bronze/users/")
df_users.show(5, truncate=False)

df_campaigns = spark.read.parquet("s3a://bronze/campaigns/")
df_campaigns.show(5, truncate=False)

df_conversions = spark.read.parquet("s3a://bronze/conversions/")
df_conversions.show(5, truncate=False)

df_impressions = spark.read.parquet("s3a://bronze/impressions/")
df_impressions.show(5, truncate=False)

df_clicks = spark.read.parquet("s3a://bronze/clicks/")
df_clicks.show(50, truncate=False)

df_users_silver = spark.read.parquet("s3a://silver/users/")
df_users_silver.show(5, truncate=False)

df_campaigns_silver = spark.read.parquet("s3a://silver/campaigns/")
df_campaigns_silver.show(5, truncate=False)

df_users_impressions = spark.read.parquet("s3a://silver/impressions/")
df_users_impressions.show(5, truncate=False)

df_users_clicks = spark.read.parquet("s3a://silver/clicks/")
df_users_clicks.show(5, truncate=False)

df_users_conversions = spark.read.parquet("s3a://silver/conversions/")
df_users_conversions.show(5, truncate=False)