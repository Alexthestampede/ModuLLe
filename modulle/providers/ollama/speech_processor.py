"""Generic text-to-speech with Ollama.

Wraps Ollama's OpenAI-compatible ``/v1/audio/speech`` endpoint (Ollama >=
0.40.9) so applications can turn text into spoken audio. All application
logic should be built on top of this generic interface.

Speech requires a speech-capable model, e.g. ``phonon`` (31M English
checkpoint from Gradium). Note that ``phonon`` requires Apple's MLX runtime,
so it currently runs only on Apple Silicon hosts.
"""

import io
import re
import wave
from pathlib import Path
from typing import List, Optional

from modulle.utils.logging_config import get_logger

from .client import OllamaClient

logger = get_logger(__name__.replace("modulle.providers.", ""))

# Ollama's speech endpoint accepts at most 4096 characters per request.
MAX_INPUT_CHARS = 4096

# Split sentences after ., ! or ? followed by whitespace (keeps the mark)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_text(text: str, limit: int = MAX_INPUT_CHARS) -> List[str]:
    """
    Split text into chunks of at most ``limit`` characters.

    Prefers paragraph breaks, then sentence boundaries, then word
    boundaries (hard-splitting only pathological long words).

    Args:
        text: Text to split
        limit: Maximum characters per chunk (default: 4096)

    Returns:
        List of non-empty chunks, in order
    """
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    if len(text) <= limit:
        return [text] if text else []

    segments: List[str] = []
    for paragraph in text.split("\n\n"):
        if len(paragraph) <= limit:
            segments.append(paragraph)
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
            if len(sentence) <= limit:
                segments.append(sentence)
            else:
                segments.extend(_split_words(sentence, limit))

    chunks: List[str] = []
    current = ""
    for segment in segments:
        candidate = f"{current}\n\n{segment}" if current else segment
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = segment
    if current:
        chunks.append(current)
    return chunks


def _split_words(text: str, limit: int) -> List[str]:
    """Hard-split an over-long segment at word boundaries."""
    chunks: List[str] = []
    current = ""
    for word in text.split():
        while len(word) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(word[:limit])
            word = word[limit:]
        candidate = f"{current} {word}".strip()
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def _riff_chunks(data: bytes):
    """Iterate (chunk_id, body) pairs in a RIFF/WAVE file."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = data[pos + 8 : pos + 8 + size]
        yield chunk_id, body
        pos += 8 + size + (size & 1)


def _wav_frames(data: bytes) -> bytes:
    """Return the sample frames of a WAV file."""
    for chunk_id, body in _riff_chunks(data):
        if chunk_id == b"data":
            return body
    raise ValueError("WAV file has no data chunk")


def _wav_params(data: bytes):
    """Return (nchannels, sampwidth, framerate) from a WAV file."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            return wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
    except wave.Error as e:
        raise ValueError(f"not a WAV file: {e}") from e


def stitch_wav(parts: List[bytes]) -> bytes:
    """
    Concatenate WAV files into one, rewriting a valid header.

    Ollama writes WAV headers whose size fields hold the maximum value
    (the length is unknown while streaming), so the stitched output is
    re-encoded to carry correct sizes.

    Args:
        parts: WAV audio chunks in order (same sample format required)

    Returns:
        A single WAV file as bytes
    """
    if not parts:
        return b""
    nchannels, sampwidth, framerate = _wav_params(parts[0])
    frames = b"".join(_wav_frames(part) for part in parts)

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(frames)
    return out.getvalue()


class OllamaSpeechProcessor:
    """
    Generic text-to-speech processor using Ollama.

    Applications build voice features by passing the text they want spoken.
    Long text is automatically split into <= 4096-character requests and
    the audio chunks are stitched back together.
    """

    def __init__(
        self,
        model: str = "lfm3-omni",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout: Optional[int] = None,
        client: Optional[OllamaClient] = None,
    ):
        """
        Initialize Ollama speech processor.

        Args:
            model: Speech model name (e.g. 'phonon')
            base_url: Ollama server URL (optional, uses default if not provided)
            api_key: Ollama Cloud API key (https://ollama.com only; optional)
            request_timeout: Timeout in seconds for API requests
                (optional, uses config default)
            client: Optional pre-built OllamaClient (mainly for testing)
        """
        self.model = model
        if client is not None:
            self.client = client
        elif base_url or api_key:
            self.client = OllamaClient(
                base_url=base_url, request_timeout=request_timeout, api_key=api_key
            )
        else:
            self.client = OllamaClient(request_timeout=request_timeout)
        logger.info(f"Ollama speech processor initialized with model: {model}")

    def speak(
        self,
        text: str,
        voice: str = "default",
        response_format: str = "wav",
        output_path: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Convert text to speech.

        Args:
            text: Text to speak (any length; split automatically)
            voice: Voice name (case-insensitive); 'default' uses the
                checkpoint's own voice
            response_format: 'wav' (default; 16-bit mono 24 kHz) or 'pcm'
            output_path: Optional file path to also write the audio to

        Returns:
            Audio bytes, or None on error / empty input.

        Example:
            >>> processor.speak("Hello from ModuLLe", voice="Zoey")
            >>> processor.speak("Save me", output_path="speech.wav")
        """
        if not text or not text.strip():
            logger.warning("Nothing to speak: text is empty")
            return None
        if response_format not in ("wav", "pcm"):
            raise ValueError("response_format must be 'wav' or 'pcm'")

        chunks = split_text(text)
        parts: List[bytes] = []
        for chunk in chunks:
            part = self.client.speech(
                model=self.model,
                text=chunk,
                voice=voice,
                response_format=response_format,
            )
            if part is None:
                return None
            parts.append(part)

        if response_format == "wav":
            audio = stitch_wav(parts)
        else:
            audio = b"".join(parts)

        if output_path:
            Path(output_path).write_bytes(audio)
            logger.info(f"Wrote {len(audio)} bytes of speech audio to {output_path}")

        return audio

    def speak_to_file(
        self,
        text: str,
        path: str,
        voice: str = "default",
        response_format: str = "wav",
    ) -> Optional[str]:
        """
        Convert text to speech and write it to a file.

        Args:
            text: Text to speak
            path: Destination file path (e.g. 'speech.wav')
            voice: Voice name (case-insensitive)
            response_format: 'wav' (default) or 'pcm'

        Returns:
            The written path, or None on error.
        """
        audio = self.speak(text, voice=voice, response_format=response_format)
        if audio is None:
            return None
        Path(path).write_bytes(audio)
        logger.info(f"Wrote {len(audio)} bytes of speech audio to {path}")
        return str(path)

    def list_voices(self) -> List[str]:
        """
        List voices supported by the configured speech model.

        Returns:
            List of voice names, or an empty list on error.
        """
        return self.client.list_voices(self.model)


# Backward compatibility alias, matching the text/vision processors
OllamaSpeechClient = OllamaSpeechProcessor
