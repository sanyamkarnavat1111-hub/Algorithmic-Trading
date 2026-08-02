import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.database import get_connection

conn = get_connection()
try:
    with conn.cursor() as cur:
        # Check counts per timeframe
        cur.execute("SELECT timeframe, COUNT(*), MIN(open_time), MAX(open_time) FROM candles GROUP BY timeframe;")
        rows = cur.fetchall()
        print("\n=== Candles Sync Status ===")
        for r in rows:
            print(f"Timeframe: {r[0]:<4} | Count: {r[1]:<6} | Min: {r[2]} | Max: {r[3]}")
        
        # Check Postgres Database Size on disk
        # pg_database_size returns size in bytes
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()));")
        db_size = cur.fetchone()[0]
        
        # Check size per table
        cur.execute("""
            SELECT 
                relname AS table_name,
                pg_size_pretty(pg_total_relation_size(class.oid)) AS total_size
            FROM pg_class class
            JOIN pg_namespace ns ON ns.oid = class.relnamespace
            WHERE nspname = 'public' AND relkind = 'r'
            ORDER BY pg_total_relation_size(class.oid) DESC;
        """)
        tables = cur.fetchall()
        
        print("\n=== Database Disk Usage ===")
        print(f"Total Database Size: {db_size}")
        print("----------------------------")
        for t in tables:
            print(f"Table: {t[0]:<20} | Size: {t[1]}")
        print("============================\n")
        
finally:
    conn.close()
