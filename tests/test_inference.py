import numpy as np
from unittest.mock import patch, MagicMock
import backend.inference as inf


# -------------------------
# Helpers (mocks)
# -------------------------
class MockVectorizer:
    def transform(self, texts):
        class Dense:
            def toarray(self_inner):
                return np.array([[0.1, 0.2]])
        return Dense()

    def get_feature_names_out(self):
        return ["f1", "f2"]


class MockMLB:
    def inverse_transform(self, preds):
        return [("cs.LG",)]


# -------------------------
# initialize()
# -------------------------
@patch("backend.inference.joblib.load")
def test_initialize_success(mock_load):
    mock_load.side_effect = [MockVectorizer(), MockMLB()]

    inf.initialize()

    assert inf.STATE == "ready"
    assert inf.vectorizer is not None
    assert inf.mlb is not None


@patch("backend.inference.joblib.load")
def test_initialize_failure(mock_load):
    mock_load.side_effect = Exception("fail")

    try:
        inf.initialize()
        assert False
    except Exception:
        assert inf.STATE == "error"


# -------------------------
# call_mlflow()
# -------------------------
@patch("backend.inference.requests.post")
def test_call_mlflow_dict_response(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"predictions": [[0.9, 0.1]]}
    mock_response.raise_for_status.return_value = None

    mock_post.return_value = mock_response

    X = np.array([[1, 2]])

    preds = inf.call_mlflow(X)

    assert preds.shape == (1, 2)


@patch("backend.inference.requests.post")
def test_call_mlflow_list_response(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = [[0.9, 0.1]]
    mock_response.raise_for_status.return_value = None

    mock_post.return_value = mock_response

    X = np.array([[1, 2]])

    preds = inf.call_mlflow(X)

    assert preds.shape == (1, 2)


@patch("backend.inference.requests.post")
def test_call_mlflow_failure(mock_post):
    mock_post.side_effect = Exception("fail")

    try:
        inf.call_mlflow(np.array([[1, 2]]))
        assert False
    except RuntimeError:
        assert inf.STATE == "error"


# -------------------------
# predict_batch()
# -------------------------
def setup_ready_state():
    inf.STATE = "ready"
    inf.vectorizer = MockVectorizer()
    inf.mlb = MockMLB()


@patch("backend.inference.call_mlflow")
def test_predict_batch_success(mock_call):
    setup_ready_state()

    mock_call.return_value = np.array([[0.9]])

    result = inf.predict_batch(["test text"])

    assert isinstance(result, list)
    assert isinstance(result[0], list)


def test_predict_batch_not_ready():
    inf.STATE = "starting"

    try:
        inf.predict_batch(["test"])
        assert False
    except RuntimeError:
        assert True


def test_predict_batch_empty():
    setup_ready_state()

    result = inf.predict_batch([])

    assert result == []


# -------------------------
# predict_single()
# -------------------------
@patch("backend.inference.predict_batch")
def test_predict_single(mock_batch):
    mock_batch.return_value = [["cs.LG"]]

    result = inf.predict_single("test")

    assert result == ["cs.LG"]