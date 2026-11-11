from datetime import datetime
from typing import Literal

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from pydantic import BaseModel

Status = Literal["processing", "ready", "failed"]

COLUMNS = (
    "id::text AS id, filename, file_hash, status, error, "
    "chunk_count, created_at, updated_at"
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('processing', 'ready', 'failed')),
    error TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


class Document(BaseModel):
    id: str
    filename: str
    file_hash: str
    status: Status
    error: str | None = None
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class DocumentRepository:
    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        schema: str = "public",
    ) -> None:
        self._conn_args = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
            "connect_timeout": 5,
            "options": f"-c search_path={schema}",
        }
        self._schema = schema

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(**self._conn_args, row_factory=dict_row)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(self._schema)
                )
            )
            conn.execute(SCHEMA_SQL)

    def drop_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(self._schema)
                )
            )

    def create(self, filename: str, file_hash: str) -> Document:
        with self._connect() as conn:
            row = conn.execute(
                f"INSERT INTO documents (filename, file_hash, status) "
                f"VALUES (%s, %s, 'processing') RETURNING {COLUMNS}",
                (filename, file_hash),
            ).fetchone()
        return Document(**row)

    def get_by_hash(self, file_hash: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {COLUMNS} FROM documents WHERE file_hash = %s",
                (file_hash,),
            ).fetchone()
        return Document(**row) if row else None

    def mark_ready(self, document_id: str, chunk_count: int) -> Document:
        return self._update(
            "status = 'ready', chunk_count = %s, error = NULL",
            (chunk_count,),
            document_id,
        )

    def mark_failed(self, document_id: str, error: str) -> Document:
        return self._update("status = 'failed', error = %s", (error,), document_id)
    
    def mark_processing(self, document_id: str) -> Document:
        return self._update("status = 'processing', error = NULL", (), document_id)

    def _update(self, assignments: str, values: tuple, document_id: str) -> Document:
        with self._connect() as conn:
            row = conn.execute(
                f"UPDATE documents SET {assignments}, updated_at = now() "
                f"WHERE id = %s::uuid RETURNING {COLUMNS}",
                (*values, document_id),
            ).fetchone()
        if row is None:
            raise LookupError(f"document {document_id} not found")
        return Document(**row)