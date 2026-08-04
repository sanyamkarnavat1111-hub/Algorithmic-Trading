"""
check_db_sizes.py — Compare row counts between Local and Render databases.
"""

import os
import psycopg2
from dotenv import load_dotenv

def check_sizes():
    load_dotenv(override=True)
    
    local_url = os.environ.get("LOCAL_DB_URL")
    render_url = os.environ.get("RENDER_DB_EXTERNAL_URL")
    
    if not local_url or not render_url:
        print("Missing LOCAL_DB_URL or RENDER_DB_EXTERNAL_URL in .env")
        return
        
    tables = [
        "candles", 
        "scalers", 
        "model_store", 
        "portfolio", 
        "trades", 
        "predictions_log", 
        "app_logs"
    ]
    
    print("\n--- Database Size & Row Count Comparison ---")
    print(f"{'Table':<18} | {'Local DB (Rows / Size)':<26} | {'Render DB (Rows / Size)':<26}")
    print("-" * 75)
    
    try:
        # Connect to both databases
        l_conn = psycopg2.connect(local_url)
        r_conn = psycopg2.connect(render_url)
        
        with l_conn.cursor() as l_cur, r_conn.cursor() as r_cur:
            for table in tables:
                # Get Local count & size
                try:
                    l_cur.execute(f"SELECT COUNT(*), pg_size_pretty(pg_total_relation_size('{table}')) FROM {table}")
                    row = l_cur.fetchone()
                    l_str = f"{row[0]} rows / {row[1]}"
                except Exception:
                    l_conn.rollback()
                    l_str = "Not created"
                    
                # Get Render count & size
                try:
                    r_cur.execute(f"SELECT COUNT(*), pg_size_pretty(pg_total_relation_size('{table}')) FROM {table}")
                    row = r_cur.fetchone()
                    r_str = f"{row[0]} rows / {row[1]}"
                except Exception:
                    r_conn.rollback()
                    r_str = "Not created"
                    
                print(f"{table:<18} | {l_str:<26} | {r_str:<26}")
                
    except Exception as e:
        print(f"\n❌ Connection error: {e}")
    finally:
        if 'l_conn' in locals(): l_conn.close()
        if 'r_conn' in locals(): r_conn.close()

if __name__ == "__main__":
    check_sizes()
