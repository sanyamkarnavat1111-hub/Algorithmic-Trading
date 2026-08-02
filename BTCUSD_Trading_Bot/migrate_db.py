"""
migrate_db.py — Push local PostgreSQL data to Render PostgreSQL

This script copies the following tables from your local DB to your Render DB:
- candles
- model_store
- model_versions

It uses the connection strings from your .env file:
- LOCAL_DB_URL
- RENDER_DB_EXTERNAL_URL
"""

import sys
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Add project root to path so we can import data.database
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# We expect LOCAL_DB_URL to be defined in .env, otherwise we use standard localhost
LOCAL_URL = os.getenv("LOCAL_DB_URL", "postgresql://postgres:postgres@localhost:5432/trading_bot")
RENDER_URL = os.getenv("RENDER_DB_EXTERNAL_URL")

if not RENDER_URL:
    print("❌ ERROR: RENDER_DB_EXTERNAL_URL not found in .env file.")
    print("Please add your Render External Database URL to your .env file and try again.")
    sys.exit(1)

def migrate_table(table_name, local_conn, render_conn):
    print(f"Migrating table '{table_name}'...")
    
    # 1. Fetch data from local
    with local_conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {table_name}")
        rows = cur.fetchall()
        
        if not rows:
            print(f"  - No data found in local '{table_name}'. Skipping.")
            return

        # Get column names
        colnames = [desc[0] for desc in cur.description]

    print(f"  - Found {len(rows)} rows. Uploading to Render...")
    
    # 2. Insert data into Render
    with render_conn.cursor() as cur:
        # Clear the target table first
        cur.execute(f"TRUNCATE TABLE {table_name} CASCADE")
        
        # Build query (e.g. INSERT INTO candles (col1, col2) VALUES %s)
        cols_str = ",".join(colnames)
        query = f"INSERT INTO {table_name} ({cols_str}) VALUES %s"
        
        # Use execute_values for fast batch insertion
        psycopg2.extras.execute_values(cur, query, rows, page_size=1000)
    
    render_conn.commit()
    print(f"  ✅ '{table_name}' migration complete!")

def run_migration():
    print("Connecting to local database...")
    try:
        local_conn = psycopg2.connect(LOCAL_URL)
    except Exception as e:
        print(f"❌ Failed to connect to local database: {e}")
        return
    
    print("Connecting to Render database...")
    try:
        render_conn = psycopg2.connect(RENDER_URL)
    except Exception as e:
        print(f"❌ Failed to connect to Render database: {e}")
        return

    print("\n--- Starting Data Migration ---\n")
    
    tables_to_migrate = [
        "candles",
        "model_store",
        "model_versions",
        "scalers",
        "trades"
    ]

    for table in tables_to_migrate:
        migrate_table(table, local_conn, render_conn)

    print("\n🎉 Migration finished successfully! Your Render database is fully populated.")
    
    local_conn.close()
    render_conn.close()

if __name__ == "__main__":
    run_migration()
