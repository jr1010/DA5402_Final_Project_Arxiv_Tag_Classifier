from unittest.mock import patch


def test_root(client):
    res = client.get("/")
    assert res.status_code == 200


@patch("backend.main.inference.predict_single")
def test_predict_success(mock_predict, client):
    mock_predict.return_value = ["cs.lg"]

    res = client.post("/predict", json={"text": "Neural networks"})

    assert res.status_code == 200
    data = res.json()

    assert "labels" in data
    assert data["labels"] == ["cs.lg"]


def test_predict_empty(client):
    res = client.post("/predict", json={"text": ""})

    # Your API currently allows empty input and returns 200
    assert res.status_code == 200


def test_predict_no_body(client):
    res = client.post("/predict", json={})
    assert res.status_code == 422