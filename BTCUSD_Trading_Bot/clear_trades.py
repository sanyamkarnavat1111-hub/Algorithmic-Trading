"""Clear trades, portfolio, and predictions_log from Render DB."""
import psycopg2

url = "postgresql://algo_trading_db_9v7v_user:E6DXv7PttPz9k83N0Nt8cbXqtJhP7yue@dpg-d9p0osajnfac73bud5s0-a.frankfurt-postgres.render.com/algo_trading_db_9v7"

print("Connecting...")
conn = psycopg2.connect(url, connect_timeout=15)
try:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM trades")
        cur.execute("DELETE FROM portfolio")
        cur.execute("DELETE FROM predictions_log")
        cur.execute("DELETE FROM app_logs")
    conn.commit()
    print("Done. All trades, portfolio, predictions_log, and app_logs cleared.")
finally:
    conn.close()
