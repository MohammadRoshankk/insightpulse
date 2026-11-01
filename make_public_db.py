import os
import sqlite3
import pandas as pd

SRC_DB = "insightpulse.db"
DST_DIR = "data"
DST_DB = os.path.join(DST_DIR, "public_reviews.db")

def make_public_db():
    os.makedirs(DST_DIR, exist_ok=True)
    if os.path.exists(DST_DB):
        os.remove(DST_DB)

    # Read reviews from insightpulse.db
    src_conn = sqlite3.connect(SRC_DB)
    df_reviews = pd.read_sql_query("SELECT id, text, rating, date, sentiment, category FROM reviews", src_conn)
    src_conn.close()

    # Write only reviews table to data/public_reviews.db
    dst_conn = sqlite3.connect(DST_DB)
    df_reviews.to_sql("reviews", dst_conn, if_exists="replace", index=False)
    
    # Verify row count
    cur = dst_conn.cursor()
    cur.execute("SELECT count(*) FROM reviews")
    count = cur.fetchone()[0]
    dst_conn.close()

    print(f"data/public_reviews.db created successfully with {count} rows.")

if __name__ == "__main__":
    make_public_db()

