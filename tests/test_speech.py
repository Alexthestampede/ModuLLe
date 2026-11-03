"""
Tests for text-to-speech support (Ollama /v1/audio/speech).
"""

import io
import wave
from unittest.mock import MagicMock, patch

import pytest

from modulle import create_speech_processor
from modulle.providers.ollama.client import OllamaClient
from modulle.providers.ollama.speech_processor import (
    MAX_INPUT_CHARS,
    OllamaSpeechProcessor,
    split_text,
    stitch_wav,
)


def _wav_bytes(frames: bytes = b"\x00\x01" * 10, rate: int = 24000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames)
    return out.getvalue()


def _frames_of(data: bytes) -> bytes:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _ok(content: bytes):
    response = MagicMock()
    response.ok = True
    response.status_code = 200
    response.content = content
    return response


def _err(message: str, status: int = 400):
    response = MagicMock()
    response.ok = False
    response.status_code = status
    response.content = b""
    response.json.return_value = {"error": {"message": message, "type": "invalid_request_error"}}
    return response


class TestClientSpeech:
    def test_posts_expected_payload_and_returns_bytes(self):
        audio = _wav_bytes()
        client = OllamaClient(base_url="http://localhost:11434")
        with patch(
            "modulle.providers.ollama.client.requests.post",
            return_value=_ok(audio),
        ) as mock_post:
            result = client.speech(
                model="phonon", text="Hello", voice="Zoey", response_format="wav"
            )

        assert result == audio
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:11434/v1/audio/speech"
        sent = kwargs["json"]
        assert sent == {
            "model": "phonon",
            "input": "Hello",
            "voice": "Zoey",
            "response_format": "wav",
        }

    def test_pcm_format_passthrough(self):
        raw = b"\x01\x02\x03\x04"
        client = OllamaClient(base_url="http://localhost:11434")
        with patch(
            "modulle.providers.ollama.client.requests.post",
            return_value=_ok(raw),
        ):
            result = client.speech(model="phonon", text="x", response_format="pcm")
        assert result == raw

    def test_rejects_unknown_format(self):
        client = OllamaClient()
        with pytest.raises(ValueError, match="wav"):
            client.speech(model="phonon", text="x", response_format="mp3")

    def test_structured_error_returns_none(self):
        client = OllamaClient()
        with patch(
            "modulle.providers.ollama.client.requests.post",
            return_value=_err('"ornith" does not support speech'),
        ):
            assert client.speech(model="ornith", text="x") is None

    def test_network_error_returns_none(self):
        import requests

        client = OllamaClient()
        with patch(
            "modulle.providers.ollama.client.requests.post",
            side_effect=requests.exceptions.ConnectionError("boom"),
        ):
            assert client.speech(model="phonon", text="x") is None

    def test_api_key_header_sent(self):
        client = OllamaClient(base_url="https://ollama.com", api_key="k")
        with patch(
            "modulle.providers.ollama.client.requests.post",
            return_value=_ok(b"a"),
        ) as mock_post:
            client.speech(model="phonon", text="x")
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer k"

    def test_list_voices_via_api_show(self):
        client = OllamaClient()
        with patch(
            "modulle.providers.ollama.client.requests.post",
            return_value=MagicMock(
                ok=True,
                status_code=200,
                json=MagicMock(
                    return_value={
                        "capabilities": ["speech"],
                        "voices": ["default", "Zoey", "Easton", "Freya"],
                    }
                ),
            ),
        ) as mock_post:
            voices = client.list_voices("phonon")
        assert voices == ["default", "Zoey", "Easton", "Freya"]
        assert mock_post.call_args.args[0] == "http://localhost:11434/api/show"

    def test_list_voices_error_returns_empty(self):
        client = OllamaClient()
        with patch(
            "modulle.providers.ollama.client.requests.post",
            return_value=_err("model not found", status=404),
        ):
            assert client.list_voices("phonon") == []


class TestSplitText:
    def test_short_text_single_chunk(self):
        assert split_text("Hello.") == ["Hello."]

    def test_empty_text(self):
        assert split_text("") == []

    def test_splits_on_sentences(self):
        text = " ".join(["This is sentence number %d." % i for i in range(400)])
        chunks = split_text(text, limit=200)
        assert chunks
        assert all(len(c) <= 200 for c in chunks)
        assert " ".join(chunks).replace("\n\n", " ").split() == text.split()

    def test_never_exceeds_limit(self):
        text = "x" * 10 + " " + ("word " * 2000)
        chunks = split_text(text, limit=500)
        assert all(len(c) <= 500 for c in chunks)

    def test_default_limit(self):
        text = "a " * (MAX_INPUT_CHARS // 2 + 100)
        chunks = split_text(text)
        assert all(len(c) <= MAX_INPUT_CHARS for c in chunks)


class TestStitchWav:
    def test_concatenates_frames(self):
        a = _wav_bytes(b"\x01\x00" * 4)
        b = _wav_bytes(b"\x02\x00" * 6)
        out = stitch_wav([a, b])
        assert _frames_of(out) == _frames_of(a) + _frames_of(b)
        with wave.open(io.BytesIO(out), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000

    def test_handles_max_size_headers(self):
        # Ollama streams WAVs whose header size fields hold the maximum value
        a = bytearray(_wav_bytes(b"\x01\x00" * 3))
        a[40:44] = (0xFFFFFFFF).to_bytes(4, "little")  # data chunk size
        a[4:8] = (0xFFFFFFFF).to_bytes(4, "little")  # RIFF size
        out = stitch_wav([bytes(a)])
        assert _frames_of(out) == b"\x01\x00" * 3

    def test_empty(self):
        assert stitch_wav([]) == b""

    def test_non_wav_rejected(self):
        with pytest.raises(ValueError):
            stitch_wav([b"not a wav"])


class TestSpeechProcessor:
    def test_speak_returns_wav(self):
        processor = OllamaSpeechProcessor(model="phonon", client=MagicMock())
        processor.client.speech.return_value = _wav_bytes(b"\x03\x00" * 5)
        audio = processor.speak("Hello")
        assert _frames_of(audio) == b"\x03\x00" * 5
        processor.client.speech.assert_called_once()

    def test_long_text_is_chunked_and_stitched(self):
        processor = OllamaSpeechProcessor(model="phonon", client=MagicMock())
        processor.client.speech.side_effect = [
            _wav_bytes(b"\x01\x00" * 3),
            _wav_bytes(b"\x02\x00" * 2),
        ]
        long_text = ("A sentence. " * 400).strip()
        audio = processor.speak(long_text)
        assert processor.client.speech.call_count == 2
        assert _frames_of(audio) == b"\x01\x00" * 3 + b"\x02\x00" * 2
        for call in processor.client.speech.call_args_list:
            assert len(call.kwargs["text"]) <= MAX_INPUT_CHARS

    def test_pcm_chunks_concatenated(self):
        processor = OllamaSpeechProcessor(model="phonon", client=MagicMock())
        processor.client.speech.side_effect = [b"\x01\x00", b"\x02\x00"]
        long_text = ("A sentence. " * 400).strip()
        audio = processor.speak(long_text, response_format="pcm")
        assert audio == b"\x01\x00\x02\x00"

    def test_speak_to_file(self, tmp_path):
        processor = OllamaSpeechProcessor(model="phonon", client=MagicMock())
        processor.client.speech.return_value = _wav_bytes(b"\x04\x00" * 2)
        path = tmp_path / "speech.wav"
        returned = processor.speak_to_file("Hi", str(path))
        assert returned == str(path)
        assert path.read_bytes() == stitch_wav([processor.client.speech.return_value])

    def test_empty_text_returns_none(self):
        processor = OllamaSpeechProcessor(model="phonon", client=MagicMock())
        assert processor.speak("   ") is None
        processor.client.speech.assert_not_called()

    def test_error_propagates_as_none(self):
        processor = OllamaSpeechProcessor(model="phonon", client=MagicMock())
        processor.client.speech.return_value = None
        assert processor.speak("Hello") is None

    def test_list_voices_delegates(self):
        processor = OllamaSpeechProcessor(model="phonon", client=MagicMock())
        processor.client.list_voices.return_value = ["default", "Zoey"]
        assert processor.list_voices() == ["default", "Zoey"]
        processor.client.list_voices.assert_called_once_with("phonon")


class TestFactorySpeech:
    def teardown_method(self):
        patch.stopall()

    def test_ollama_uses_config_base_url(self):
        with patch("modulle.providers.ollama.speech_processor.OllamaClient") as mock_client_cls:
            processor = create_speech_processor("ollama", model="phonon")
        assert processor.model == "phonon"
        assert mock_client_cls.call_args.kwargs["base_url"] == "http://localhost:11434"

    def test_alias_and_default_model(self):
        with patch("modulle.factory.config.OLLAMA_SPEECH_MODEL", "voice-model"):
            with patch("modulle.providers.ollama.speech_processor.OllamaClient"):
                processor = create_speech_processor("ollamacloud", api_key="k")
        assert processor.model == "voice-model"

    def test_cloud_requires_key(self):
        with patch("modulle.factory.config.OLLAMA_API_KEY", ""):
            with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
                create_speech_processor("ollama_cloud")

    def test_unsupported_provider(self):
        with pytest.raises(ValueError, match="Ollama only"):
            create_speech_processor("openai")
