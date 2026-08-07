import os
import psycopg2
from dotenv import load_dotenv

load_dotenv(override=True)

db_url = os.environ.get("RENDER_DB_EXTERNAL_URL") or os.environ.get("LOCAL_DB_URL")
if not db_url:
    print("No DB URL")
    exit()

print(f"Connecting to {db_url.split('@')[-1]}...")
conn = psycopg2.connect(db_url)
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM predictions_log;")
print(f"Total predictions: {cur.fetchone()[0]}")

cur.execute("SELECT direction, COUNT(*) FROM predictions_log GROUP BY direction;")
print(f"Predictions breakdown:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

cur.execute("SELECT COUNT(*) FROM positions;")
print(f"Total positions: {cur.fetchone()[0]}")

cur.execute("SELECT status, COUNT(*) FROM positions GROUP BY status;")
print(f"Positions breakdown:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

cur.close()
conn.close()
