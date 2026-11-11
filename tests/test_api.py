from fastapi.testclient import TestClient

from src.hospital_rag.api import create_app
from src.hospital_rag.config import Settings, get_settings


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