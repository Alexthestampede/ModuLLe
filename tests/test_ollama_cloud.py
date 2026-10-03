"""
Tests for Ollama Cloud support (Bearer auth + factory wiring).
"""

from unittest.mock import MagicMock, patch

import pytest

from modulle import create_ai_client
from modulle.providers.ollama.client import OllamaClient


class TestOllamaClientAuth:
    """Auth header behaviour on the shared OllamaClient."""

    def test_no_key_no_header(self):
        client = OllamaClient(base_url="http://localhost:11434")
        assert client._headers == {}

    def test_key_sets_bearer_header(self):
        client = OllamaClient(base_url="https://ollama.com", api_key="sekret")
        assert client._headers == {"Authorization": "Bearer sekret"}

    def test_auth_header_sent_on_generate_and_chat(self):
        client = OllamaClient(base_url="https://ollama.com", api_key="sekret")
        payload_gen = {"model": "m", "answers": {"a": {"type": "noul", "noul": 0.5}}}
        with patch(
            "modulle.providers.ollama.client.requests.post", return_value=_ok(payload_gen)
        ) as mock_post:
            client.generate(model="m", prompt="hi")
            sent_headers = mock_post.call_args.kwargs["headers"]
            assert sent_headers["Authorization"] == "Bearer sekret"

    def test_auth_header_sent_on_list_models(self):
        client = OllamaClient(base_url="https://ollama.com", api_key="sekret")
        with patch(
            "modulle.providers.ollama.client.requests.get",
            return_value=_ok({"models": [{"name": "glm-5.3-flash"}]}),
        ) as mock_get:
            models = client.list_models()
        assert models == ["glm-5.3-flash"]
        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer sekret"

    def test_local_client_unchanged_default_url(self):
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client._headers == {}


def _ok(payload):
    response = MagicMock()
    response.ok = True
    response.status_code = 200
    response.json.return_value = payload
    return response


class TestFactoryOllamaCloud:
    """Factory wiring for the ollama_cloud provider."""

    def _patch_ollama_classes(self):
        patches = [
            patch("modulle.providers.ollama.client.OllamaClient", return_value=MagicMock()),
            patch(
                "modulle.providers.ollama.text_processor.OllamaTextClient", return_value=MagicMock()
            ),
            patch(
                "modulle.providers.ollama.vision_processor.OllamaVisionClient",
                return_value=MagicMock(),
            ),
        ]
        return [p.start() for p in patches], [p for p in patches]

    def teardown_method(self):
        patch.stopall()

    def test_creates_clients_with_api_key(self):
        mocks, _ = self._patch_ollama_classes()
        _client, text_processor, _vision = create_ai_client(
            provider="ollama_cloud", text_model="glm-5.3-flash", api_key="sekret"
        )
        assert mocks[1].call_args.kwargs["api_key"] == "sekret"
        assert mocks[1].call_args.kwargs["base_url"] == "https://ollama.com"

    def test_no_key_raises_value_error(self):
        with patch("modulle.factory.config.OLLAMA_API_KEY", ""):
            with pytest.raises(ValueError, match="ollama.com/settings/keys"):
                create_ai_client(provider="ollama_cloud")

    def test_alias_accepted(self):
        with patch("modulle.providers.ollama.client.OllamaClient", return_value=MagicMock()):
            with patch(
                "modulle.providers.ollama.text_processor.OllamaTextClient", return_value=MagicMock()
            ):
                with patch(
                    "modulle.providers.ollama.vision_processor.OllamaVisionClient",
                    return_value=MagicMock(),
                ):
                    client, _tp, _vp = create_ai_client(provider="ollamacloud", api_key="k")
        assert client is not None

    def test_defaults_from_config(self):
        mocks, _ = self._patch_ollama_classes()
        with patch("modulle.factory.config.OLLAMA_CLOUD_TEXT_MODEL", "some-model"):
            with patch("modulle.factory.config.OLLAMA_API_KEY", "cfgkey"):
                create_ai_client(provider="ollama_cloud")
        assert mocks[1].call_args.kwargs["model"] == "some-model"
        assert mocks[1].call_args.kwargs["api_key"] == "cfgkey"

    def test_no_vision_model_means_none(self):
        with patch("modulle.factory.config.OLLAMA_CLOUD_VISION_MODEL", ""):
            with patch("modulle.providers.ollama.client.OllamaClient", return_value=MagicMock()):
                with patch(
                    "modulle.providers.ollama.text_processor.OllamaTextClient",
                    return_value=MagicMock(),
                ):
                    _c, _tp, vp = create_ai_client(provider="ollama_cloud", api_key="k")
        assert vp is None


class TestChatWithToolsCloudContent:
    """Regression: cloud tool-call responses with null/whitespace content."""

    def test_whitespace_content_with_tool_calls(self):
        client = OllamaClient(base_url="https://ollama.com", api_key="k")
        payload = {
            "message": {
                "role": "assistant",
                "content": "\n",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "add", "arguments": {"a": 1}}}
                ],
            }
        }
        with patch(
            "modulle.providers.ollama.client.requests.post",
            return_value=_ok(payload),
        ):
            result = client.chat_with_tools(
                model="m", messages=[{"role": "user", "content": "hi"}], tools=[]
            )
        assert result["finish_reason"] == "tool_calls"
        assert result["content"] is None
        assert result["tool_calls"][0]["name"] == "add"

    def test_null_content_with_tool_calls(self):
        client = OllamaClient(base_url="https://ollama.com", api_key="k")
        payload = {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "add", "arguments": {"a": 1}}}],
            }
        }
        with patch(
            "modulle.providers.ollama.client.requests.post",
            return_value=_ok(payload),
        ):
            result = client.chat_with_tools(
                model="m", messages=[{"role": "user", "content": "hi"}], tools=[]
            )
        assert result["finish_reason"] == "tool_calls"

    def test_whitespace_content_no_tools(self):
        client = OllamaClient(base_url="https://ollama.com", api_key="k")
        payload = {"message": {"role": "assistant", "content": "\n"}}
        with patch(
            "modulle.providers.ollama.client.requests.post",
            return_value=_ok(payload),
        ):
            result = client.chat_with_tools(
                model="m", messages=[{"role": "user", "content": "hi"}], tools=[]
            )
        assert result["finish_reason"] == "stop"
        assert result["content"] is None
