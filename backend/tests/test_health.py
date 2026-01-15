"""Tests for the health endpoint."""


def test_health_check(client):
    """Test that the health endpoint returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
