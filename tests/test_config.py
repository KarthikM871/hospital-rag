import pytest
from pydantic import ValidationError

from src.hospital_rag.config import Settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("RAG_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("RAG_LOG_LEVEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.chunk_size == 500
    assert settings.log_level == "INFO"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "800")

    settings = Settings(_env_file=None)

    assert settings.chunk_size == 800
    assert isinstance(settings.chunk_size, int)


def test_invalid_value_fails(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "abc")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)