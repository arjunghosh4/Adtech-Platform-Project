from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("KafkaToMinIOAllTopics") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

def read_kafka_topic(topic, output_path):
    df = spark.read \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:29092") \
        .option("subscribe", topic) \
        .option("startingOffsets", "earliest") \
        .option("endingOffsets", "latest") \
        .load()

    df_parsed = df.selectExpr("CAST(value AS STRING) as json_record")

    df_parsed.write.mode("overwrite").parquet(output_path)
    print(f"✅ Finished writing {topic} to {output_path}")

# Run for all topics
read_kafka_topic("users", "s3a://bronze/users/")
read_kafka_topic("campaigns", "s3a://bronze/campaigns/")
read_kafka_topic("impressions", "s3a://bronze/impressions/")
read_kafka_topic("clicks", "s3a://bronze/clicks/")
read_kafka_topic("conversions", "s3a://bronze/conversions/")

spark.stop()