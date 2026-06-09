from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app_factory import create_app


def test_create_app_starts_and_health_works():
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert app.title == "Proactive AI"
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
