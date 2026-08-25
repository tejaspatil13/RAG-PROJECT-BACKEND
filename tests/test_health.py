import os

# Prevent external API configuration from being required during CI import.
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}