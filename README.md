# ModuLLe - Modular LLM Provider Abstraction Layer

A clean, generic Python interface for multiple AI/LLM providers. Write your code once, switch providers anytime.

**Philosophy**: ModuLLe provides the AI interface, **you** provide the intelligence through prompts.

## 🎯 What is ModuLLe?

ModuLLe is a **pure abstraction layer** - it gives you generic `generate()`, `chat()`, and `analyze_image()` methods that work across all AI providers. No opinionated logic, no hardcoded use cases - just clean, flexible AI access.

You build your application logic (summarization, translation, classification, etc.) by crafting the right prompts.

## ✨ Supported Providers

### Local (Free, Private)
- **Ollama** - Run LLMs locally
- **LM Studio** - User-friendly local LLM server

### Cloud (API-based)
- **OpenAI** - GPT-4o, GPT-4o-mini, etc.
- **Google Gemini** - Gemini 1.5 Flash/Pro
- **Anthropic Claude** - Claude 3.5 Haiku/Sonnet

## 🚀 Quick Start

### Installation

```bash
# Basic (local providers)
pip install -e .

# With cloud providers
pip install -e ".[openai]"   # For OpenAI
pip install -e ".[gemini]"   # For Google Gemini
pip install -e ".[claude]"   # For Anthropic Claude
pip install -e ".[all]"      # For all providers
```

### First-Time Setup

Run the interactive configuration wizard:

```bash
modulle-config
```

This will guide you through:
1. Selecting your AI provider (Ollama, LM Studio, OpenAI, Gemini, or Claude)
2. Testing connections and listing available models
3. Configuring API keys (for cloud providers)
4. Saving your settings to `~/.modulle.json`

### Basic Usage

```python
from modulle import create_ai_client

# Create a text processor (works with any provider!)
_, text_processor, _ = create_ai_client(
    provider='ollama',  # or 'openai', 'gemini', 'claude', 'lm_studio'
    text_model='llama2'
)

# Use generic generate() - YOU craft the prompts
summary = text_processor.generate(
    prompt="Summarize this text in 2 sentences: [your text here]",
    system_prompt="You are a professional summarizer",
    temperature=0.3
)

# Or use chat() for multi-turn conversations
response = text_processor.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7
)
```

## 🔧 API Reference

ModuLLe provides **3 simple methods**:

### 1. `generate(prompt, system_prompt, temperature, max_tokens)`
Generic text generation - use for any single-turn task

```python
# Summarization
result = processor.generate(
    prompt=f"Summarize: {text}",
    system_prompt="You are a professional summarizer",
    temperature=0.3
)

# Translation
result = processor.generate(
    prompt=f"Translate to Spanish: {text}",
    temperature=0.2
)

# Classification
result = processor.generate(
    prompt=f"Is this positive or negative? {text}",
    system_prompt="You are a sentiment analyzer. Answer only: positive or negative",
    temperature=0.1
)
```

### 2. `chat(messages, temperature, max_tokens)`
Multi-turn conversations

```python
response = processor.chat(
    messages=[
        {"role": "system", "content": "You are an expert programmer"},
        {"role": "user", "content": "How do I reverse a string in Python?"},
        {"role": "assistant", "content": "You can use string[::-1]"},
        {"role": "user", "content": "What about in JavaScript?"}
    ],
    temperature=0.5
)
```

### 3. `analyze_image(image_data, prompt, temperature, max_tokens)`
Generic image analysis

```python
result = vision_processor.analyze_image(
    image_data=base64_encoded_image,
    prompt="Describe this image in detail",
    temperature=0.7
)

# Or any other vision task
result = vision_processor.analyze_image(
    image_data=base64_encoded_image,
    prompt="Extract all text from this image",  # OCR
    temperature=0.1
)
```

## 💡 Philosophy: Build Your Logic with Prompts

ModuLLe doesn't include methods like `detect_clickbait()` or `generate_title()` - instead, you build these using `generate()`:

```python
# Clickbait detection - YOU define what "clickbait" means
is_clickbait = processor.generate(
    prompt=f"Is this title clickbait? Answer yes or no.\n\nTitle: {title}",
    system_prompt="You detect sensationalized headlines",
    temperature=0.1
)

# Title generation - YOU control the style
title = processor.generate(
    prompt=f"Generate a professional title for this summary: {summary}",
    system_prompt="You write clear, informative titles (max 80 chars)",
    temperature=0.2
)
```

See `examples/article_summarizer.py` for a complete example of building article processing on top of ModuLLe.

## 🔀 Provider Switching

Change providers by changing **one parameter**:

```python
# Development: Use free local Ollama
_, processor, _ = create_ai_client('ollama', text_model='llama2')

# Production: Switch to OpenAI (same API!)
_, processor, _ = create_ai_client('openai', text_model='gpt-4o-mini', api_key='...')

# Cost optimization: Use Gemini's free tier
_, processor, _ = create_ai_client('gemini', text_model='gemini-1.5-flash', api_key='...')
```

## 📁 Project Structure

```
modulle/
├── base.py              # Abstract base classes
├── factory.py           # create_ai_client() factory
├── config.py            # Provider configurations
├── providers/           # 5 provider implementations
│   ├── ollama/
│   ├── lm_studio/
│   ├── openai/
│   ├── gemini/
│   └── claude/
└── utils/               # Logging, HTTP client
```

## 🎓 Examples

- **`examples/basic_usage.py`** - Simple text generation examples
- **`examples/article_summarizer.py`** - Complete article processing app built on ModuLLe
- **`examples/multi_provider.py`** - Switching between providers
- **`examples/multi_language.py`** - Multi-language text generation

## ⚙️ Configuration

ModuLLe supports three configuration methods with priority order:

1. **User config file** (`~/.modulle.json`) - Highest priority
2. **Environment variables** - Medium priority
3. **Code defaults** - Lowest priority

### Interactive Configuration Wizard (Recommended)

Run the interactive setup wizard to configure ModuLLe:

```bash
# First-time setup
modulle-config

# Or with Python
python -m modulle.cli.config_wizard
```

The wizard will:
- Help you choose an AI provider (Ollama, LM Studio, OpenAI, Gemini, or Claude)
- Test connections to local servers
- List available models and help you select them
- Configure API keys for cloud providers
- Save settings to `~/.modulle.json`

### User Configuration File

Create `~/.modulle.json` in your home directory:

```json
{
  "provider": "ollama",
  "ollama_base_url": "http://localhost:11434",
  "ollama_text_model": "llama2",
  "ollama_vision_model": "llava"
}
```

See `.modulle.json.example` in the repository for a complete example with all providers.

**Benefits:**
- Easy to edit and version control
- No need to set environment variables
- Persists across sessions
- Can be shared across projects

### Environment Variables

```bash
# Ollama (local)
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_TEXT_MODEL="llama2"

# OpenAI (cloud)
export OPENAI_API_KEY="your-api-key"
export OPENAI_TEXT_MODEL="gpt-4o-mini"

# Gemini (cloud)
export GEMINI_API_KEY="your-api-key"
export GEMINI_TEXT_MODEL="gemini-1.5-flash"

# Claude (cloud)
export ANTHROPIC_API_KEY="your-api-key"
export CLAUDE_TEXT_MODEL="claude-3-5-haiku-20241022"
```

### Python Configuration

```python
from modulle import create_ai_client

# Override defaults in code
_, processor, _ = create_ai_client(
    provider='ollama',
    text_model='mistral',  # Different model
    base_url='http://custom-server:11434'
)
```

### Inspecting Ollama Models

List available Ollama models and their capabilities:

```bash
# Inspect models on default server
modulle-inspect-ollama

# Or specify a custom server
modulle-inspect-ollama http://192.168.1.100:11434

# Or with Python
python -m modulle.cli.ollama_inspector
```

This shows:
- Model names and sizes
- Parameter counts
- Context window sizes
- Vision capabilities
- Model families and formats

## 🧪 Testing

```bash
# Quick structure test
python3 test_basic.py

# Run examples
cd examples
python basic_usage.py
python article_summarizer.py
```

## 📊 Provider Comparison

| Feature | Ollama | LM Studio | OpenAI | Gemini | Claude |
|---------|--------|-----------|--------|--------|--------|
| **Cost** | Free | Free | Paid | Free tier + Paid | Paid |
| **Privacy** | ✅ Local | ✅ Local | ❌ Cloud | ❌ Cloud | ❌ Cloud |
| **Internet** | Setup only | Setup only | Required | Required | Required |
| **Best For** | Privacy, dev | Beginners | Production | Cost-effective | Premium quality |

## 🔒 Privacy

**Local providers (Ollama, LM Studio)**:
- All processing on your machine
- No data sent to external servers
- Works offline after model download

**Cloud providers (OpenAI, Gemini, Claude)**:
- Data sent to provider's servers
- Subject to provider's privacy policy
- Internet connection required

## 📚 Documentation

- **README.md** (this file) - API reference
- **REFACTORING.md** - Design philosophy and refactoring details
- **examples/README.md** - Example usage guide
- **examples/article_summarizer.py** - Complete application example

## 🤝 Contributing

ModuLLe is designed to be a minimal, generic abstraction layer. Contributions should:
- Keep the core generic (no domain-specific logic)
- Maintain consistency across all providers
- Add functionality through examples, not core library

## 📝 License

MIT License - Free for personal and commercial use

## 🎯 Key Benefits

1. **Generic** - No opinionated logic, works for any application
2. **Simple** - Only 3 methods: `generate()`, `chat()`, `analyze_image()`
3. **Flexible** - Build any feature through prompts
4. **Provider-agnostic** - Switch providers with one parameter
5. **Clean** - Minimal dependencies, focused scope
6. **Production-ready** - Error handling, retries, logging

## 🚀 Getting Started

1. **Install**: `pip install -e .`
2. **Configure**: `modulle-config` (interactive wizard)
3. **Test**: `python3 test_basic.py`
4. **Try examples**: `cd examples && python basic_usage.py`
5. **Read**: `examples/article_summarizer.py` - See how to build real apps
6. **Build**: Use ModuLLe's generic API to create your application!

---

**ModuLLe**: Clean abstractions for AI providers. You provide the prompts, we provide the interface. 🚀
