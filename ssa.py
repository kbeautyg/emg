#!/usr/bin/env python3
"""
fill_embeddings.py  —  one-shot loader for Supabase pgvector columns
author: KBeautyGuide / ChatGPT

env vars required:
  SUPA_HOST     = db.<project>.supabase.co   (без порта, SSL)
  SUPA_DB       = postgres
  SUPA_USER     = postgres
  SUPA_PASS     = *****
  OPENAI_KEY    = sk-q46eUGZRTFDbzFFjZrCloA
"""

import os, json, time, textwrap, requests, psycopg
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

PG_DSN = f"host={os.getenv('SUPA_HOST')} port={os.getenv('SUPA_PORT')} dbname={os.getenv('SUPA_DB')} user={os.getenv('SUPA_USER')} password={os.getenv('SUPA_PASS')}"
OPENAI_URL = "https://hubai.loe.gg/v1/embeddings"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_KEY')}"
}

TABLES = ["korakora", "origaship"]          # ← можно расширить
BATCH  = 96                                 # сколько текстов за один запрос к OpenAI
MODEL  = "text-embedding-ada-002"

def fetch_unembedded(cur, table, limit=1000):
    sql = textwrap.dedent(f"""
        SELECT id,
               CONCAT_WS(' ', name, brand, product_types, active_ingredients) AS content
        FROM   public.{table}
        WHERE  embedding IS NULL
          AND  LENGTH(CONCAT_WS(' ', name, brand, product_types, active_ingredients)) > 5
        LIMIT  {limit};
    """)
    cur.execute(sql)
    return cur.fetchall()

def embed_batch(texts):
    body = {"model": MODEL, "input": texts}
    r = requests.post(OPENAI_URL, headers=HEADERS, data=json.dumps(body), timeout=60)
    r.raise_for_status()
    # ответ формата {"data":[{"embedding":[..],"index":0}, ...]}
    return [item["embedding"] for item in r.json()["data"]]

def save_embeddings(cur, table, rows, vecs):
    # pgvector формат — строка вида '[0.12,0.34,...]'
    sql = f"UPDATE public.{table} SET embedding = %s::vector WHERE id = %s"
    data = [(json.dumps(v), row[0]) for row, v in zip(rows, vecs)]
    cur.executemany(sql, data)

def main():
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        for table in TABLES:
            print(f"\n⏳ Processing table: {table}")
            while True:
                rows = fetch_unembedded(cur, table, limit=BATCH)
                if not rows:
                    print("✓ done")
                    break
                texts = [row[1] for row in rows]
                vecs  = embed_batch(texts)
                save_embeddings(cur, table, rows, vecs)
                conn.commit()
                time.sleep(0.5)       # бережём прокси
                tqdm.write(f"  +{len(rows)} rows updated")

if __name__ == "__main__":
    main()
