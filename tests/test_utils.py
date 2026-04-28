import os
import tempfile
from backend.utils import load_config


def test_load_config_valid():
    # create a temporary yaml file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp.write("key: value\nnumber: 42")
        tmp_path = tmp.name

    config = load_config(tmp_path)

    assert config["key"] == "value"
    assert config["number"] == 42

    os.remove(tmp_path)


def test_load_config_invalid_path():
    try:
        load_config("non_existent.yaml")
        assert False  # should not reach here
    except FileNotFoundError:
        assert True


def test_load_config_default():
    # only works if backend/config.yaml exists
    config = load_config()
    assert isinstance(config, dict)