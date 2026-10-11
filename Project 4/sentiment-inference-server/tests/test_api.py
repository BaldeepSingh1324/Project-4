"""API tests. The real model is replaced with a stub so tests run offline and fast."""
import pytest
from fastapi.testclient import TestClient

import app.main as main


def fake_model(texts):
    return [{"label": "POSITIVE" if "good" in t else "NEGATIVE", "score": 0.99} for t in texts]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda: fake_model)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict(client):
    r = client.post("/predict", json={"text": "this is good"})
    assert r.status_code == 200
    assert r.json()["label"] == "POSITIVE"


def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"texts": ["good", "bad"]})
    assert [p["label"] for p in r.json()["predictions"]] == ["POSITIVE", "NEGATIVE"]


def test_batch_too_large(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_BATCH_SIZE", 2)
    assert client.post("/predict/batch", json={"texts": ["a", "b", "c"]}).status_code == 413
