# ModuLLe - Generic AI Provider Abstraction Layer

**Version 0.2.0** - A thin, generic abstraction layer for AI/LLM providers

## What is ModuLLe?

ModuLLe provides a **unified, generic interface** for multiple AI providers. It gives you simple `generate()` and `chat()` methods that work identically across:

- **Ollama** (local open-source models)
- **LM Studio** (local models with OpenAI-compatible API)
- **OpenAI** (GPT-4, GPT-3.5, etc.)
- **Claude** (Anthropic)
- **Gemini** (Google)

## Design Philosophy

**ModuLLe is intentionally generic.**

- ✓ Provides ONLY: `generate()`, `chat()`, `analyze_image()`
- ✓ NO application-specific methods (no "summarize", no "detect_clickbait", etc.)
- ✓ YOU build your logic by crafting prompts
- ✓ Works for ANY application: chatbots, summarizers, translators, code analyzers, etc.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Text Generation

```python
from modulle.providers.ollama.text_processor import OllamaTextProcessor

# Initialize processor
processor = OllamaTextProcessor(model='llama2')

# Generate text
result = processor.generate(
    prompt="Explain quantum computing in simple terms",
    system_prompt="You are a helpful teacher",
    temperature=0.7
)

print(result)
```

### Multi-turn Chat

```python
response = processor.chat(
    messages=[
        {"role": "system", "content": "You are a Python expert"},
        {"role": "user", "content": "How do I read a file?"},
        {"role": "assistant", "content": "Use open() function"},
        {"role": "user", "content": "Show me an example"}
    ],
    temperature=0.5
)

print(response)
```

### Vision Analysis

```python
from modulle.providers.openai.vision_processor import OpenAIVisionProcessor
import base64

# Initialize vision processor
vision = OpenAIVisionProcessor(model='gpt-4o')

# Analyze image
with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

description = vision.analyze_image(
    image_data=image_data,
    prompt="Describe what you see in this image",
    temperature=0.3
)

print(description)
```

## Provider Examples

### Ollama (Local)

```python
from modulle.providers.ollama.text_processor import OllamaTextProcessor

processor = OllamaTextProcessor(
    model='llama2',  # or 'mistral', 'codellama', etc.
    base_url='http://localhost:11434'  # optional
)
```

### OpenAI (Cloud)

```python
from modulle.providers.openai.text_processor import OpenAITextProcessor

processor = OpenAITextProcessor(
    model='gpt-4o-mini',  # or 'gpt-4o', 'gpt-3.5-turbo'
    api_key='sk-...'  # or set OPENAI_API_KEY env var
)
```

### Claude (Cloud)

```python
from modulle.providers.claude.text_processor import ClaudeTextProcessor

processor = ClaudeTextProcessor(
    model='claude-3-5-haiku-20241022',
    api_key='sk-ant-...'  # or set ANTHROPIC_API_KEY env var
)
```

### Gemini (Cloud)

```python
from modulle.providers.gemini.text_processor import GeminiTextProcessor

processor = GeminiTextProcessor(
    model='gemini-1.5-flash',
    api_key='...'  # or set GEMINI_API_KEY env var
)
```

### LM Studio (Local)

```python
from modulle.providers.lm_studio.text_processor import LMStudioTextProcessor

processor = LMStudioTextProcessor(
    model='local-model',
    base_url='http://localhost:1234'  # optional
)
```

## Building Application Logic

ModuLLe is intentionally generic. You build your application logic by crafting prompts.

### Example: Text Summarization

```python
def summarize(text, max_words=100):
    return processor.generate(
        prompt=f"Summarize this text in {max_words} words or less:\n\n{text}",
        system_prompt="You are a professional summarizer",
        temperature=0.3
    )
```

### Example: Translation

```python
def translate(text, target_language):
    return processor.generate(
        prompt=f"Translate to {target_language}:\n\n{text}",
        system_prompt="You are a professional translator",
        temperature=0.2
    )
```

### Example: Sentiment Analysis

```python
def analyze_sentiment(text):
    return processor.generate(
        prompt=f"What is the sentiment of this text? Answer: positive/negative/neutral\n\n{text}",
        system_prompt="You are a sentiment analysis expert",
        temperature=0.1
    )
```

### Example: Code Review

```python
def review_code(code):
    return processor.generate(
        prompt=f"Review this code for bugs and improvements:\n\n```\n{code}\n```",
        system_prompt="You are an expert code reviewer",
        temperature=0.5
    )
```

## Complete Example: Article Summarizer

See `examples/article_summarizer.py` for a full implementation showing:
- Clickbait detection
- Summary generation with different strategies
- Title generation
- Complete orchestration

This demonstrates how to build complex application logic on top of ModuLLe's simple generic API.

## API Reference

### BaseTextProcessor

#### `generate(prompt, system_prompt=None, temperature=0.7, max_tokens=None)`

Generate text from a single prompt.

**Parameters:**
- `prompt` (str): User prompt/instruction
- `system_prompt` (str, optional): System prompt to set model behavior
- `temperature` (float): Sampling temperature (0.0=deterministic, 1.0+=creative)
- `max_tokens` (int, optional): Maximum tokens to generate

**Returns:** Generated text string or None on error

#### `chat(messages, temperature=0.7, max_tokens=None)`

Multi-turn conversation.

**Parameters:**
- `messages` (List[Dict]): Conversation history
  - Each dict has `role` (system/user/assistant) and `content`
- `temperature` (float): Sampling temperature
- `max_tokens` (int, optional): Maximum tokens to generate

**Returns:** Generated response string or None on error

### BaseVisionProcessor

#### `analyze_image(image_data, prompt, temperature=0.7, max_tokens=None)`

Analyze an image with a custom prompt.

**Parameters:**
- `image_data` (str): Base64-encoded image data
- `prompt` (str): Instruction describing what to analyze
- `temperature` (float): Sampling temperature
- `max_tokens` (int, optional): Maximum tokens to generate

**Returns:** Analysis result string or None on error

## Examples

Run the examples:

```bash
# Basic usage (generic methods)
python examples/basic_usage.py

# Article summarizer (building app logic)
python examples/article_summarizer.py

# Multi-provider comparison
python examples/multi_provider.py
```

## Configuration

Set via environment variables:

```bash
# Ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_TEXT_MODEL=llama2

# OpenAI
export OPENAI_API_KEY=sk-...
export OPENAI_TEXT_MODEL=gpt-4o-mini

# Claude
export ANTHROPIC_API_KEY=sk-ant-...
export CLAUDE_TEXT_MODEL=claude-3-5-haiku-20241022

# Gemini
export GEMINI_API_KEY=...
export GEMINI_TEXT_MODEL=gemini-1.5-flash

# LM Studio
export LM_STUDIO_BASE_URL=http://localhost:1234
export LM_STUDIO_TEXT_MODEL=local-model
```

## Why ModuLLe?

### Before ModuLLe
```python
# Different API for each provider
ollama_client = OllamaAPI(...)
ollama_client.generate(...)

openai_client = OpenAI(...)
openai_client.chat.completions.create(...)

claude_client = Anthropic(...)
claude_client.messages.create(...)
```

### With ModuLLe
```python
# Same API for all providers
processor = OllamaTextProcessor(...)
result = processor.generate(...)

processor = OpenAITextProcessor(...)
result = processor.generate(...)  # Identical API!

processor = ClaudeTextProcessor(...)
result = processor.generate(...)  # Identical API!
```

## Use Cases

ModuLLe works for ANY application:

- **Chatbots**: Use `chat()` for conversations
- **Content Generation**: Use `generate()` with creative prompts
- **Data Analysis**: Use `generate()` with analytical prompts
- **Translation**: Use `generate()` with translation prompts
- **Summarization**: Use `generate()` with summarization prompts
- **Code Analysis**: Use `generate()` with code review prompts
- **Image Analysis**: Use `analyze_image()` for vision tasks
- **OCR**: Use `analyze_image()` with text extraction prompts
- **Classification**: Use `generate()` with classification prompts
- **And more**: If you can prompt it, you can build it!

## Migration from v0.1.x

See `REFACTORING.md` for detailed migration guide.

**Summary:**
- Old methods (`detect_clickbait()`, `generate_summary()`, etc.) are removed
- Use generic `generate()` and `chat()` methods instead
- See `examples/article_summarizer.py` for how to rebuild article logic

## Contributing

ModuLLe should remain **intentionally simple and generic**:

- ✓ Add new provider implementations
- ✓ Fix bugs in existing providers
- ✓ Improve error handling
- ✗ Don't add application-specific methods
- ✗ Don't add hardcoded prompts to the library

Application logic belongs in YOUR code, not in ModuLLe.

## License

[Your License Here]

## Links

- Examples: `examples/`
- Refactoring Guide: `REFACTORING.md`
- API Documentation: `modulle/base.py`

---

**ModuLLe: Simple, Generic, Powerful**
