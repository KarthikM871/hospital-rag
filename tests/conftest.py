import uuid

import psycopg
import pytest

from src.hospital_rag.config import get_settings
from src.hospital_rag.documents import DocumentRepository


@pytest.fixture
def repo():
    settings = get_settings()
    if not settings.postgres_password:
        pytest.skip("RAG_POSTGRES_PASSWORD is not set in .env")
    repository = DocumentRepository(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        schema=f"test_{uuid.uuid4().hex[:8]}",
    )
    try:
        repository.ensure_schema()
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres is not available: {exc}")
    yield repository
    repository.drop_schema()