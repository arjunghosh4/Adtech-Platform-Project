from pyspark.sql import SparkSession

# ----------------------------
# Spark Session with MinIO config
# ----------------------------
spark = (
    SparkSession.builder
        .appName("LoadSilverToPostgres")
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.path.style.access", True)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ----------------------------
# JDBC connection details
# ----------------------------
jdbc_url = "jdbc:postgresql://localhost:5433/ads_db"
connection_props = {
    "user": "admin",
    "password": "admin",
    "driver": "org.postgresql.Driver"
}

# ----------------------------
# Helper function
# ----------------------------
def load_to_postgres(table_name):
    df = spark.read.parquet(f"s3a://silver/{table_name}/")
    print(f"✅ Loaded {table_name} from MinIO with {df.count()} rows")

    df.write.jdbc(
        url=jdbc_url,
        table=table_name,
        mode="overwrite",
        properties=connection_props
    )
    print(f"🚀 Written {table_name} → Postgres")

# ----------------------------
# Load all silver datasets
# ----------------------------
for t in ["users", "campaigns", "impressions", "clicks", "conversions"]:
    load_to_postgres(t)

spark.stop()