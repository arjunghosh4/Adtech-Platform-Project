from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, from_json, to_date, to_timestamp
from pyspark.sql.types import MapType, StringType

# --------------------------
# Spark session (MinIO)
# --------------------------
spark = (
    SparkSession.builder
        .appName("BronzeToSilver")
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.path.style.access", True)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# --------------------------
# Spec for each table
# (type keywords are handled below)
# --------------------------
TABLE_SPECS = {
    "users": [
        ("user_id", "int"),
        ("region", "string_lower"),
        ("device", "string_lower"),
        ("subscription_tier", "string_lower"),
        ("signup_date", "date")
    ],
    "campaigns": [
        ("campaign_id", "int"),
        ("advertiser", "string_lower"),
        ("budget", "double"),
        ("target_region", "string_lower"),
        ("ad_format", "string_lower"),
        ("start_date", "date"),
        ("end_date", "date"),
    ],
    "impressions": [
        ("impression_id", "int"),
        ("user_id", "int"),
        ("campaign_id", "int"),
        ("timestamp", "timestamp"),
        ("device", "string_lower"),
        ("ad_slot", "string_lower"),
        ("ad_format", "string_lower"),
    ],
    # clicks may or may not include device/ad_slot/ad_format in your files; this tolerates both
    "clicks": [
        ("click_id", "int"),
        ("impression_id", "int"),
        ("user_id", "int"),
        ("campaign_id", "int"),
        ("timestamp", "timestamp"),
        ("click_position", "int"),
        ("device", "string_lower_optional"),
        ("ad_slot", "string_lower_optional"),
        ("ad_format", "string_lower_optional"),
    ],
    # conversions may contain impression_id/timestamp/device/ad_slot/ad_format in your files; optional is fine
    "conversions": [
        ("conversion_id", "int"),
        ("click_id", "int"),
        ("user_id", "int"),
        ("campaign_id", "int"),
        ("order_id", "string"),              # keep UUID case
        ("product_category", "string_lower"),
        ("revenue", "double"),
        ("impression_id", "int_optional"),
        ("timestamp", "timestamp_optional"),
        ("device", "string_lower_optional"),
        ("ad_slot", "string_lower_optional"),
        ("ad_format", "string_lower_optional"),
        ("click_position", "int_optional"),
    ],
}

def cast_expr(mapcol, name, kind):
    base = mapcol.getItem(name)  # null if key not present
    if kind == "int":
        return base.cast("int").alias(name)
    if kind == "double":
        return base.cast("double").alias(name)
    if kind == "date":
        return to_date(base).alias(name)
    if kind == "timestamp":
        return to_timestamp(base).alias(name)
    if kind == "string_lower":
        return lower(trim(base)).alias(name)
    if kind == "string":
        return trim(base).alias(name)

    # optional variants – same cast, null is fine if missing
    if kind == "int_optional":
        return base.cast("int").alias(name)
    if kind == "timestamp_optional":
        return to_timestamp(base).alias(name)
    if kind == "string_lower_optional":
        return lower(trim(base)).alias(name)

    # default: plain string
    return trim(base).alias(name)

def clean_strings(df):
    # trim + lowercase for string-typed cols (already done in cast_expr for selected fields)
    for c, t in df.dtypes:
        if t == "string":
            df = df.withColumn(c, trim(col(c)))
    return df

def process_table(table):
    raw = spark.read.parquet(f"s3a://bronze/{table}/")
    parsed = raw.select(from_json(col("json_record"), MapType(StringType(), StringType())).alias("data"))

    # select+cast columns per spec (missing keys -> null)
    cols = [cast_expr(col("data"), name, kind) for (name, kind) in TABLE_SPECS[table]]
    df = parsed.select(*cols)

    df = clean_strings(df).dropDuplicates()

    # Light QA: forbid negative budget/revenue; coerce to null if somehow bad
    if table == "campaigns":
        df = df.withColumn("budget", col("budget").cast("double"))
    if table == "conversions":
        df = df.withColumn("revenue", col("revenue").cast("double"))

    df.write.mode("overwrite").parquet(f"s3a://silver/{table}/")
    print(f"✅ Silver: {table} written")

for t in ["users", "campaigns", "impressions", "clicks", "conversions"]:
    process_table(t)

print("\n🎉 Silver layer successfully created (schema-robust & typed)!\n")