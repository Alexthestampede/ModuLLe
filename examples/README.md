# ModuLLe Examples

This directory contains example scripts demonstrating how to use ModuLLe.

## Prerequisites

Before running these examples, make sure you have:

1. Installed ModuLLe:
   ```bash
   pip install -e ..
   ```

2. Set up at least one AI provider:
   - **Ollama**: Install from https://ollama.ai and run `ollama serve`
   - **LM Studio**: Install from https://lmstudio.ai and start the server
   - **Cloud providers**: Get API keys from OpenAI, Google, or Anthropic

## Examples

### basic_usage.py

Demonstrates basic text summarization using the Ollama provider.

**Usage:**
```bash
# Make sure Ollama is running
ollama serve

# Run the example
python basic_usage.py
```

**What it does:**
- Creates an Ollama client
- Checks server health
- Lists available models
- Generates a summary with title and clickbait detection

### multi_provider.py

Shows how to use the same code with different AI providers.

**Usage:**
```bash
# Edit the file to uncomment providers you want to test
# Add your API keys for cloud providers

python multi_provider.py
```

**What it does:**
- Tests multiple providers with the same code
- Demonstrates provider-agnostic API
- Compares results from different providers

### multi_language.py

Demonstrates generating summaries in different languages.

**Usage:**
```bash
python multi_language.py
```

**What it does:**
- Generates summaries in 8 different languages
- Shows multilingual capabilities
- Demonstrates language parameter usage

## Customization

All examples can be easily modified:

1. **Change the provider:**
   ```python
   create_ai_client(provider='openai', api_key='your-key')
   ```

2. **Use different models:**
   ```python
   create_ai_client(provider='ollama', text_model='mistral')
   ```

3. **Adjust parameters:**
   ```python
   generate_summary(text=text, language="Spanish", max_length=300)
   ```

## Troubleshooting

### "Connection refused" errors
- Make sure your local server (Ollama/LM Studio) is running
- Check that the base_url matches your server configuration

### "API key not found" errors
- Set environment variables or pass api_key parameter
- Verify your API key is valid

### "Model not found" errors
- For Ollama: Run `ollama pull model-name` first
- For LM Studio: Make sure model is loaded in the GUI

### Slow response times
- Local models: Check your GPU/CPU resources
- Cloud providers: Check your internet connection
- Try smaller models for faster responses

## Next Steps

After trying these examples:

1. Read the main [README.md](../README.md) for full API documentation
2. Check the [modulle/](../modulle/) source code for implementation details
3. Build your own application using ModuLLe!
