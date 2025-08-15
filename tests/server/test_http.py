"""Tests for the AutoIntent FastAPI HTTP API."""

import os
import tempfile
from pathlib import Path

import pytest

from autointent import Pipeline
from autointent.configs import LoggingConfig
from tests.conftest import get_search_space

# Skip all tests in this file if FastAPI dependencies are not available
fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture
def trained_pipeline_path(dataset):
    """Train a pipeline and save it to a temporary directory."""
    # Create temporary directory for this test
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "test_api"

        # Get a light search space for faster testing
        search_space = get_search_space("light")

        # Create and train pipeline
        pipeline_optimizer = Pipeline.from_search_space(search_space)
        logging_config = LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True)
        pipeline_optimizer.set_config(logging_config)

        # Train and save pipeline
        context = pipeline_optimizer.fit(dataset)
        context.dump()

        yield logging_config.dirpath


@pytest.fixture
def api_client(trained_pipeline_path):
    """Create a FastAPI test client with the trained pipeline."""
    # Set environment variable for the API
    original_path = os.environ.get("AUTOINTENT_PATH")
    os.environ["AUTOINTENT_PATH"] = str(trained_pipeline_path)

    try:
        # Import after setting environment variable to ensure it's picked up
        from autointent.server.http import app

        with TestClient(app) as client:
            yield client
    finally:
        # Restore original environment
        if original_path is not None:
            os.environ["AUTOINTENT_PATH"] = original_path
        else:
            os.environ.pop("AUTOINTENT_PATH", None)


def test_health_endpoint(api_client):
    """Test the health check endpoint."""
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_single_utterance(api_client):
    """Test the predict endpoint with a single utterance."""
    response = api_client.post("/predict", json={"utterances": ["hello world"]})

    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 1
    # Single-label classification - can be int (in-scope) or None (out-of-scope)
    assert data["predictions"][0] is None or isinstance(data["predictions"][0], int)


def test_predict_endpoint_multiple_utterances(api_client):
    """Test the predict endpoint with multiple utterances."""
    test_utterances = ["hello world", "book a flight", "what's the weather like"]

    response = api_client.post("/predict", json={"utterances": test_utterances})

    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == len(test_utterances)

    # Check that all predictions are integers (in-scope) or None (out-of-scope)
    for prediction in data["predictions"]:
        assert prediction is None or isinstance(prediction, int)


def test_predict_endpoint_empty_utterances(api_client):
    """Test the predict endpoint with empty utterances list."""
    response = api_client.post("/predict", json={"utterances": []})

    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert data["predictions"] == []
