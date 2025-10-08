import psycopg2

# Connect to Postgres
conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="ads_db",
    user="admin",
    password="admin"
)

cur = conn.cursor()

# Fetch all tables in the 'public' schema
cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name;
""")
tables = [r[0] for r in cur.fetchall()]

# Print columns for each table
print("\n🧱 DATABASE SCHEMA OVERVIEW\n")
for t in tables:
    cur.execute(f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position;
    """, (t,))
    cols = cur.fetchall()
    print(f"📂 {t}")
    for c in cols:
        print(f"   - {c[0]} ({c[1]})")
    print("")

cur.close()
conn.close()