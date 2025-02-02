from unittest.mock import MagicMock, patch

import pytest

from autointent.generation.utterances.generator import Generator
from autointent.generation.utterances.schemas import Message


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")


@pytest.fixture
def mock_openai_client():
    with patch("openai.OpenAI") as mock_client:
        yield mock_client


def test_generator_initialization(mock_openai_client):
    generator = Generator()
    assert generator.client == mock_openai_client.return_value
    assert generator.model_name == "gpt-3.5-turbo"


def test_get_chat_completion(mock_openai_client):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    mock_openai_client.return_value.chat.completions.create.return_value = mock_response

    generator = Generator()
    messages = [Message(role="user", content="Hello")]
    response = generator.get_chat_completion(messages)

    assert response == "Test response"
    mock_openai_client.return_value.chat.completions.create.assert_called_once()
