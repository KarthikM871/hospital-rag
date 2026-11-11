import time

import psycopg

from src.hospital_rag.config import get_settings

s = get_settings()
for i in range(10):
    start = time.time()
    conn = psycopg.connect(
        host=s.postgres_host,
        port=s.postgres_port,
        dbname=s.postgres_db,
        user=s.postgres_user,
        password=s.postgres_password,
        connect_timeout=5,
    )
    conn.execute("select 1")
    conn.close()
    print(i, round(time.time() - start, 2), "s", flush=True)