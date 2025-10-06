#!/usr/bin/env python3
import os, json, requests
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values

# --------- ENV ---------
SUPABASE_HOST = os.environ["SUPABASE_HOST"]      # e.g. db.xxxxx.supabase.co
SUPABASE_DB   = os.environ.get("SUPABASE_DB","postgres")
SUPABASE_USER = os.environ["SUPABASE_USER"]      # usually: postgres
SUPABASE_PASS = os.environ["SUPABASE_PASS"]      # service role password
SUPABASE_PORT = int(os.environ.get("SUPABASE_PORT","5432"))
GEMINI_KEY    = os.environ["GEMINI_API_KEY"]

DATA_DIR      = Path("data/parsed")
EMBED_URL     = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"

BATCH_SIZE = 50   # tune for speed vs. rate limits

# --------- HELPERS ---------
def embed(text: str):
    r = requests.post(
        f"{EMBED_URL}?key={GEMINI_KEY}",
        json={"content": {"parts": [{"text": text[:8000]}]}},
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    return j["embedding"]["values"]  # list[float], length should be 768

def to_pgvector_literal(vec):
    # pgvector accepts string literal: '[v1,v2,...]'
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

def upsert_rows(cur, rows):
    # rows is a list of tuples:
    # (chunk_id, paper_id, section, para_idx, text, vector_literal_str)
    execute_values(
        cur,
        """
        INSERT INTO chunks (chunk_id, paper_id, section, para_idx, text, embedding)
        VALUES %s
        ON CONFLICT (chunk_id) DO UPDATE SET
          section = EXCLUDED.section,
          para_idx = EXCLUDED.para_idx,
          text = EXCLUDED.text,
          embedding = EXCLUDED.embedding
        """,
        rows,
        template="(%s,%s,%s,%s,%s,%s::vector)",
        page_size=len(rows),
    )

# --------- MAIN ---------
def main():
    conn = psycopg2.connect(
        host=SUPABASE_HOST,
        dbname=SUPABASE_DB,
        user=SUPABASE_USER,
        password=SUPABASE_PASS,
        port=SUPABASE_PORT,
        sslmode="require",
    )
    cur = conn.cursor()

    files = sorted(DATA_DIR.glob("*.jsonl"))
    if not files:
        print("No parsed files found in data/parsed/*.jsonl — run 01_make_chunks.py first.")
        return

    for p in files:
        print(f"→ Embedding & upserting: {p.name}")
        rows_batch = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                # get embedding
                vec = embed(rec["text"])
                vec_lit = to_pgvector_literal(vec)

                rows_batch.append((
                    rec["chunk_id"],
                    rec["paper_id"],
                    rec["section"],
                    rec["para_idx"],
                    rec["text"],
                    vec_lit
                ))

                if len(rows_batch) >= BATCH_SIZE:
                    upsert_rows(cur, rows_batch)
                    conn.commit()
                    rows_batch.clear()

        if rows_batch:
            upsert_rows(cur, rows_batch)
            conn.commit()

        print(f"✓ Done: {p.name}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()