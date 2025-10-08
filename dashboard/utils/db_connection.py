from sqlalchemy import create_engine
import pandas as pd

def get_connection():
    return create_engine("postgresql://admin:admin@localhost:5433/ads_db")

def load_table(table_name):
    engine = get_connection()
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)