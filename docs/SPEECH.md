# Speech (Text-to-Speech)

ModuLLe can turn text into spoken audio through Ollama's OpenAI-compatible
`/v1/audio/speech` endpoint (Ollama **>= 0.40.9**). Speech is Ollama-specific:
use `create_speech_processor()` or `OllamaSpeechProcessor` directly.

## Quick start

```python
from modulle import create_speech_processor

speech = create_speech_processor("ollama", model="lfm3-omni")

# Generate audio in memory (WAV bytes)
audio = speech.speak("Hello from ModuLLe.", voice="default")

# Or write straight to a file
speech.speak_to_file("Ciao, questa e' una prova.", "speech.wav", voice="female-italian")

# List the model's voices
print(speech.list_voices())
```

Set the server/model via arguments or configuration:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_SPEECH_MODEL="lfm3-omni"   # or "phonon"
```

## Models

| Model | Notes |
|-------|-------|
| `lfm3-omni` | 2B omni-modal model; has the `speech` capability, runs on CPU/GPU (llamacpp). 39 voices across many languages. |
| `phonon` | Dedicated Gradium TTS (31M). **Requires Apple's MLX runtime**, i.e. Apple Silicon. |

A speech-capable model's capabilities are exposed by `/api/show`
(`capabilities` includes `speech`, and `voices` lists the supported voices).
`list_voices()` wraps this.

**Speech models only generate speech** — chat/completion/embedding requests
to them return a `400` error, and using a non-speech model here returns
`"<model>" does not support speech`.

## Long text

The endpoint accepts at most **4,096 characters per request**. `speak()` splits
longer text automatically at paragraph, sentence, or word boundaries and
stitches the audio back together:

- `wav`: chunks are merged and the header is rewritten with correct sizes
  (Ollama's streaming WAV headers hold the maximum value in their size fields).
- `pcm`: raw 16-bit little-endian samples are concatenated.

## Formats and limits

| `response_format` | Output |
|-------------------|--------|
| `wav` (default) | WAV file: 16-bit, mono, 24 kHz |
| `pcm` | Raw 16-bit little-endian samples, mono, 24 kHz, no header |

- 4,096 characters max per request (handled automatically).
- Unsupported options (other formats, speed, `instructions`) return `400`.
- An unknown voice returns `400` and lists the available voices.
- Numbers, currency, and URLs are spelled out; emoji, bullets, and markdown
  markers are skipped. Write dates out (`October 23`, not `10/23`).

## Not supported yet

Ollama Cloud currently offers no speech models, so `create_speech_processor`
accepts `ollama_cloud` only for forward compatibility — it will fail until a
cloud TTS model exists. Other providers (OpenAI, Gemini, Claude) are not wired
up for speech; `create_speech_processor` raises `ValueError` for them.

## Example

See `examples/speech.py`:

```bash
OLLAMA_BASE_URL=http://your-server:11434 OLLAMA_SPEECH_MODEL=lfm3-omni \
  python3 examples/speech.py
```
