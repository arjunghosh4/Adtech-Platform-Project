import csv
import json
import time
from kafka import KafkaProducer

# Kafka config (connects to your Docker container)
producer = KafkaProducer(
    bootstrap_servers="localhost:29092",   # external listener
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Path to your CSV
csv_file = "data/bronze/campaigns.csv"

# Send rows from CSV to Kafka
with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Convert row (dict) to JSON and send to Kafka
        producer.send("campaigns", value=row)
        print(f"Sent: {row}")
        #time.sleep(0.5)  # simulate stream (0.5 sec delay per row)

producer.flush()
print("✅ Finished sending campaigns.csv to Kafka topic 'campaigns'")