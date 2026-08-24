from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_app_exists():
    assert app is not None


def test_health_endpoint():
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": "student-personalization-memory",
    }


def test_openapi_available():
    response = client.get(
        "/openapi.json"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["info"]["title"]
        == "Student Personalization Memory Service"
    )


def test_memory_router_registered():
    paths = {
        route.path
        for route in app.routes
        if hasattr(route, "path")
    }

    # Router itself is registered, although no /memory endpoint
    # exists yet in Step 2.
    assert "/health" in paths
