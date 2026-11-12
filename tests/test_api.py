from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.hospital_rag import ingestion
from src.hospital_rag.api import (
    create_app,
    get_embedder,
    get_llm,
    get_repo,
    get_store,
)
from src.hospital_rag.config import Settings, get_settings
from src.hospital_rag.fakes import FakeEmbedder, InMemoryVectorStore
from src.hospital_rag.interfaces import Chunk
from src.hospital_rag.parsing import ParsedPage

SOP_TEXT = "SOP-TST-001: Test\nDepartment: Lab\n1. Purpose\n" + "Purpose text. " * 20
PDF_BYTES = b"%PDF-1.4 fake pdf body"


class ReplyLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Use hand rub [1]."


@pytest.fixture
def env(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "parse_pdf",
        lambda _path: [ParsedPage(page_number=1, text=SOP_TEXT)],
    )
    app = create_app()
    store, embedder, llm = InMemoryVectorStore(), FakeEmbedder(), ReplyLLM()
    settings = Settings(
        _env_file=None,
        admin_api_key="secret",
        upload_dir=str(tmp_path),
        max_upload_mb=1,
        min_score=-1.0,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_repo] = lambda: repo
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_llm] = lambda: llm
    return SimpleNamespace(
        app=app,
        client=TestClient(app),
        store=store,
        embedder=embedder,
        llm=llm,
        repo=repo,
        tmp_path=tmp_path,
        headers={"x-api-key": "secret"},
    )


def upload(env, content=PDF_BYTES, name="sops.pdf", headers=None):
    return env.client.post(
        "/documents",
        files={"file": (name, content, "application/pdf")},
        headers=env.headers if headers is None else headers,
    )


def test_health_reports_ok_with_injected_settings():
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, app_name="test-app", environment="dev"
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "test-app",
        "environment": "dev",
    }


def test_admin_endpoints_require_the_api_key(env):
    assert upload(env, headers={}).status_code == 401
    assert upload(env, headers={"x-api-key": "wrong"}).status_code == 401
    assert env.client.get("/documents").status_code == 401


def test_admin_endpoints_fail_closed_when_key_is_not_configured(env):
    env.app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, admin_api_key=""
    )

    assert upload(env).status_code == 503


def test_rejects_non_pdf_extension(env):
    assert upload(env, content=b"hello", name="notes.txt").status_code == 415


def test_rejects_file_without_pdf_signature(env):
    assert upload(env, content=b"not a pdf", name="evil.pdf").status_code == 415


def test_rejects_oversized_file(env):
    big = b"%PDF-" + b"0" * (1024 * 1024 + 10)

    assert upload(env, content=big).status_code == 413
    assert list(env.tmp_path.glob("*.pdf")) == []


def test_upload_ingests_and_lists_document(env):
    response = upload(env)

    assert response.status_code == 201
    body = response.json()
    assert body["duplicate"] is False
    assert body["document"]["status"] == "ready"
    assert body["document"]["chunk_count"] >= 1
    assert (env.tmp_path / f"{body['document']['id']}.pdf").exists()

    listing = env.client.get("/documents", headers=env.headers).json()
    assert [d["filename"] for d in listing] == ["sops.pdf"]


def test_same_file_twice_returns_200_and_keeps_one_file(env):
    upload(env)

    second = upload(env)

    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert len(list(env.tmp_path.glob("*.pdf"))) == 1


def test_ask_returns_answer_with_exact_sources(env):
    upload(env)

    response = env.client.post("/ask", json={"question": "Purpose text."})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["answer"] == "Use hand rub [1]."
    assert body["sources"][0]["number"] == 1
    assert body["sources"][0]["sop_id"] == "SOP-TST-001"
    assert body["sources"][0]["section"] == "Purpose"
    assert body["sources"][0]["page"] == 1


def test_ask_ignores_chunks_of_documents_that_are_not_ready(env):
    orphan = env.repo.create("x.pdf", "other-hash")
    env.store.add(
        [
            Chunk(
                id="c1",
                document_id=orphan.id,
                text="orphan chunk",
                metadata={"sop_id": "SOP-X"},
            )
        ],
        env.embedder.embed(["orphan chunk"]),
    )

    body = env.client.post("/ask", json={"question": "orphan chunk"}).json()

    assert body["refused"] is True
    assert env.llm.prompts == []


def test_ask_rejects_empty_question(env):
    assert env.client.post("/ask", json={"question": ""}).status_code == 422