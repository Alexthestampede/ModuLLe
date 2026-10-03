"""
Tests for decision-model support (Ollama /v1/systemone endpoint).
"""

from unittest.mock import MagicMock, patch

import pytest

from modulle.providers.ollama.client import OllamaClient

QUESTION = {"type": "noul", "instructions": "Does the state text contain a greeting?"}


def _mock_response(payload, ok=True, status_code=200):
    response = MagicMock()
    response.ok = ok
    response.status_code = status_code
    response.json.return_value = payload
    if not ok:
        response.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return response


class TestSystemOne:
    """Tests for OllamaClient.system_one()."""

    def test_returns_full_payload(self):
        payload = {
            "model": "clef-flash",
            "answers": {"says_hello": {"type": "noul", "noul": 0.998}},
            "usage": {"input_tokens": 147, "output_tokens": 0},
        }
        with patch(
            "modulle.providers.ollama.client.requests.post", return_value=_mock_response(payload)
        ):
            result = OllamaClient(base_url="http://localhost:11434").system_one(
                model="clef-flash", state="Hello World", questions={"says_hello": QUESTION}
            )

        assert result == payload
        assert result["answers"]["says_hello"]["noul"] == pytest.approx(0.998)

    def test_posts_to_v1_systemone_with_expected_fields(self):
        payload = {"model": "clef-flash", "answers": {}, "usage": {}}
        state = {"text": "Hello World"}
        questions = {
            "says_hello": QUESTION,
            "kind": {
                "type": "choice",
                "instructions": "What is it?",
                "criteria": {"greeting": "A hello", "farewell": "A goodbye"},
            },
            "severity": {
                "type": "score",
                "instructions": "How bad?",
                "criteria": ["None", "Minor", "Major"],
            },
        }
        images = ["<base64>"]
        with patch(
            "modulle.providers.ollama.client.requests.post", return_value=_mock_response(payload)
        ) as mock_post:
            result = OllamaClient(base_url="http://localhost:11434").system_one(
                model="clef-flash",
                state=state,
                questions=questions,
                images=images,
                keep_alive="5m",
                timeout=99,
            )

        assert result == payload
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:11434/v1/systemone"
        sent = kwargs["json"]
        assert sent["model"] == "clef-flash"
        assert sent["state"] == state
        assert list(sent["questions"]) == list(questions)
        assert sent["questions"]["kind"]["type"] == "choice"
        assert sent["images"] == images
        assert sent["keep_alive"] == "5m"
        assert kwargs["timeout"] == 99

    def test_omits_optional_fields_when_unset(self):
        payload = {"model": "clef-flash", "answers": {}, "usage": {}}
        with patch(
            "modulle.providers.ollama.client.requests.post", return_value=_mock_response(payload)
        ) as mock_post:
            OllamaClient(base_url="http://localhost:11434").system_one(
                model="clef-flash", state="s", questions={"q": QUESTION}
            )

        sent = mock_post.call_args.kwargs["json"]
        assert "images" not in sent
        assert "keep_alive" not in sent

    def test_error_response_returns_none(self):
        with patch(
            "modulle.providers.ollama.client.requests.post",
            return_value=_mock_response({"error": "model not found"}, ok=False, status_code=404),
        ):
            result = OllamaClient(base_url="http://localhost:11434").system_one(
                model="clef-flash", state="s", questions={"q": QUESTION}
            )

        assert result is None

    def test_request_exception_returns_none(self):
        import requests

        with patch(
            "modulle.providers.ollama.client.requests.post",
            side_effect=requests.exceptions.ConnectionError("boom"),
        ):
            result = OllamaClient(base_url="http://localhost:11434").system_one(
                model="clef-flash", state="s", questions={"q": QUESTION}
            )

        assert result is None
