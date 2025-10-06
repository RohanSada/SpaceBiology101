#!/usr/bin/env python3
import os, json
import psycopg2
import psycopg2.extras as extras
import pandas as pd
from pathlib import Path

# ---- env (fill from your .env or export beforehand) ----
HOST = os.environ.get("SUPABASE_HOST")      # e.g. db.xxxxx.supabase.co
DB   = os.environ.get("SUPABASE_DB","postgres")
USER = os.environ.get("SUPABASE_USER","postgres")
PASS = os.environ.get("SUPABASE_PASS")      # service role password
PORT = int(os.environ.get("SUPABASE_PORT","5432"))

OUT_DIR = Path("./data/local_export"); OUT_DIR.mkdir(exist_ok=True)

BATCH = 5000  # tune to your table size

def fetch_all():
    conn = psycopg2.connect(
        host=HOST, dbname=DB, user=USER, password=PASS,
        port=PORT, sslmode="require"
    )
    cur = conn.cursor(cursor_factory=extras.DictCursor)

    # Count first
    cur.execute("SELECT count(*) FROM chunks;")
    total = cur.fetchone()[0]
    print(f"Total rows: {total}")

    offset = 0
    all_parts = []
    jsonl = (OUT_DIR / "chunks_export.jsonl").open("w", encoding="utf-8")

    while offset < total:
        cur.execute("""
          SELECT chunk_id, paper_id, section, para_idx, text, embedding::text AS embedding_text
          FROM chunks
          ORDER BY chunk_id
          OFFSET %s LIMIT %s
        """, (offset, BATCH))
        rows = cur.fetchall()
        if not rows: break

        recs = []
        for r in rows:
            emb = json.loads(r["embedding_text"]) if r["embedding_text"] else None
            rec = {
                "chunk_id": r["chunk_id"],
                "paper_id": r["paper_id"],
                "section":  r["section"],
                "para_idx": int(r["para_idx"]),
                "text":     r["text"],
                "embedding": emb
            }
            # write JSONL
            jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            recs.append(rec)

        # append to list of dataframes for Parquet
        df = pd.DataFrame.from_records(recs)
        all_parts.append(df)

        offset += len(rows)
        print(f"Downloaded {offset}/{total}")

    jsonl.close()
    cur.close(); conn.close()

    # Concatenate & write Parquet
    if all_parts:
        big = pd.concat(all_parts, ignore_index=True)
        big.to_parquet(OUT_DIR / "chunks_export.parquet", index=False)
        print("Wrote:", OUT_DIR / "chunks_export.parquet")
        print("Wrote:", OUT_DIR / "chunks_export.jsonl")

if __name__ == "__main__":
    fetch_all()