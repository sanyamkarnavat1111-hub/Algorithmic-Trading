"""
migrate_db.py — Push all data from Local PostgreSQL to Render PostgreSQL.
"""

import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

def migrate():
    load_dotenv(override=True)
    
    local_url = os.environ.get("LOCAL_DB_URL")
    render_url = os.environ.get("RENDER_DB_EXTERNAL_URL")
    
    if not local_url or not render_url:
        print("Missing LOCAL_DB_URL or RENDER_DB_EXTERNAL_URL in .env")
        return
        
    print("Connecting to local DB...")
    local_conn = psycopg2.connect(local_url)
    
    print("Connecting to Render DB...")
    render_conn = psycopg2.connect(render_url)
    
    tables = [
        "candles", 
        "scalers", 
        "model_store", 
        "portfolio", 
        "trades", 
        "predictions_log", 
        "app_logs"
    ]
    
    try:
        with local_conn.cursor() as l_cur, render_conn.cursor() as r_cur:
            print("Ensuring Render DB schemas are up to date...")
            sql = """
            CREATE TABLE IF NOT EXISTS candles (
                id          SERIAL PRIMARY KEY,
                timeframe   VARCHAR(10)   NOT NULL,
                open_time   TIMESTAMP     NOT NULL,
                open        NUMERIC(20,8) NOT NULL,
                high        NUMERIC(20,8) NOT NULL,
                low         NUMERIC(20,8) NOT NULL,
                close       NUMERIC(20,8) NOT NULL,
                volume      NUMERIC(30,8) NOT NULL,
                fetched_at  TIMESTAMP     DEFAULT NOW(),
                UNIQUE (timeframe, open_time)
            );

            CREATE TABLE IF NOT EXISTS trades (
                id              SERIAL PRIMARY KEY,
                model_id        VARCHAR(20)   NOT NULL,
                action          VARCHAR(10)   NOT NULL,
                amount_usdt     NUMERIC(20,8) NOT NULL,
                btc_quantity    NUMERIC(20,8) NOT NULL,
                price           NUMERIC(20,8) NOT NULL,
                predicted_high  NUMERIC(20,8),
                predicted_low   NUMERIC(20,8),
                direction_signal VARCHAR(10),
                confidence      NUMERIC(5,4),
                pnl             NUMERIC(20,8),
                created_at      TIMESTAMP     DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                id              SERIAL PRIMARY KEY,
                model_id        VARCHAR(20)   NOT NULL UNIQUE,
                usdt_balance    NUMERIC(20,8) NOT NULL,
                btc_quantity    NUMERIC(20,8) NOT NULL DEFAULT 0,
                btc_avg_price   NUMERIC(20,8) NOT NULL DEFAULT 0,
                updated_at      TIMESTAMP     DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS predictions_log (
                id              SERIAL PRIMARY KEY,
                model_id        VARCHAR(20)   NOT NULL,
                predicted_at    TIMESTAMP     NOT NULL DEFAULT NOW(),
                current_price   NUMERIC(20,8) NOT NULL,
                direction       VARCHAR(10)   NOT NULL,
                confidence      NUMERIC(5,4)  NOT NULL,
                predicted_high  NUMERIC(20,8) NOT NULL,
                predicted_low   NUMERIC(20,8) NOT NULL,
                actual_high     NUMERIC(20,8),
                actual_low      NUMERIC(20,8),
                was_correct     BOOLEAN
            );

            CREATE TABLE IF NOT EXISTS app_logs (
                id         SERIAL PRIMARY KEY,
                level      VARCHAR(10)  NOT NULL,
                model_id   VARCHAR(20),
                message    TEXT         NOT NULL,
                created_at TIMESTAMP    DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS scalers (
                scaler_id VARCHAR(30) PRIMARY KEY,
                scaler_blob BYTEA NOT NULL,
                updated_at TIMESTAMP DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS model_store (
                id          SERIAL PRIMARY KEY,
                timeframe   VARCHAR(10)  NOT NULL,
                model_type  VARCHAR(20)  NOT NULL,
                version     INTEGER      NOT NULL,
                model_blob  BYTEA        NOT NULL,
                accuracy    NUMERIC(10,4),
                train_rows  INTEGER,
                trained_at  TIMESTAMP    DEFAULT NOW(),
                is_active   BOOLEAN      DEFAULT FALSE,
                UNIQUE(timeframe, model_type, version)
            );
            """
            r_cur.execute(sql)
            render_conn.commit()
            
            for table in tables:
                print(f"\n--- Migrating {table} ---")
                
                # Check local count
                l_cur.execute(f"SELECT COUNT(*) FROM {table}")
                l_count = l_cur.fetchone()[0]
                
                if l_count == 0:
                    print(f"No data in {table} locally. Skipping.")
                    continue
                    
                # Check Render count
                r_cur.execute(f"SELECT COUNT(*) FROM {table}")
                r_count = r_cur.fetchone()[0]
                
                if l_count == r_count:
                    print(f"✅ Table {table} is already fully migrated ({l_count} rows). Skipping!")
                    continue
                    
                print(f"Found {l_count} rows. Pushing to Render over the network...")
                
                # Truncate this specific table on Render before inserting
                r_cur.execute(f"TRUNCATE TABLE {table} CASCADE;")
                
                # Fetch data
                l_cur.execute(f"SELECT * FROM {table}")
                rows = l_cur.fetchall()
                
                # Get column names dynamically
                col_names = [desc[0] for desc in l_cur.description]
                cols_str = ",".join(col_names)
                
                query = f"INSERT INTO {table} ({cols_str}) VALUES %s"
                
                # If table contains large binary blobs, upload 1 row at a time to prevent SSL timeout
                if table in ["model_store", "scalers"]:
                    print("  (Uploading large binary models 1-by-1...)")
                    execute_values(r_cur, query, rows, page_size=1)
                else:
                    execute_values(r_cur, query, rows, page_size=5000)
                
                # Commit after every table to save progress
                render_conn.commit()
                print(f"Successfully migrated {table}!")
                
            render_conn.commit()
            print("\n✅ Migration completed successfully! Render DB is fully synced.")
            
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        render_conn.rollback()
    finally:
        local_conn.close()
        render_conn.close()

if __name__ == "__main__":
    migrate()
