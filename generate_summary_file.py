import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

from executive_summary import batch_summarize, final_summary

SRC_DB = "insightpulse.db"
OUT_FILE = "data/executive_summary.md"

def main():
    conn = sqlite3.connect(SRC_DB)
    cur = conn.cursor()
    cur.execute("SELECT text FROM reviews WHERE text IS NOT NULL AND trim(text) != ''")
    rows = cur.fetchall()
    conn.close()

    texts = [r[0] for r in rows]
    print(f"Loaded {len(texts)} reviews from {SRC_DB}.")

    print("Generating batch summaries with Gemini...")
    batch_res = batch_summarize(texts, batch_size=50)
    
    print("Generating final executive summary...")
    summary_text = final_summary(batch_res)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n--- GENERATED EXECUTIVE SUMMARY ---\n")
    print(summary_text)
    print(f"\nWritten successfully to {OUT_FILE}")

if __name__ == "__main__":
    main()

