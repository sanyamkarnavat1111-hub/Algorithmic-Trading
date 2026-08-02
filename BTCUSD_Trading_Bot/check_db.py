import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.database import get_connection

conn = get_connection()
try:
    with conn.cursor() as cur:
        cur.execute("SELECT timeframe, COUNT(*), MIN(open_time), MAX(open_time) FROM candles GROUP BY timeframe;")
        rows = cur.fetchall()
        print("\n=== Candles in Database ===")
        for r in rows:
            print(f"Timeframe: {r[0]} | Count: {r[1]} | Min Time: {r[2]} | Max Time: {r[3]}")
        print("===========================\n")
finally:
    conn.close()
