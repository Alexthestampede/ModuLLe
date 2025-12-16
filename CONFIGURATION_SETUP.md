# ModuLLe Configuration System - Implementation Summary

This document summarizes the user-friendly configuration system added to ModuLLe.

## What Was Implemented

### 1. User Configuration File (`~/.modulle.json`)

**Location**: User's home directory (`~/.modulle.json`)

**Purpose**: Store user-specific settings that persist across sessions

**Features**:
- JSON format for easy editing
- Gitignored (user-specific, not committed to repo)
- Highest priority in configuration hierarchy
- Example file provided: `.modulle.json.example`

**Supported Settings**:
- Provider choice (`ollama`, `lm_studio`, `openai`, `gemini`, `claude`)
- Provider-specific models (text and vision)
- API keys (for cloud providers)
- Base URLs (for local providers)

### 2. Interactive Configuration Wizard

**Command**: `modulle-config`

**Location**: `modulle/cli/config_wizard.py`

**Features**:
- **Provider Selection**: Interactive menu to choose from 5 AI providers
  - Ollama (local, free)
  - LM Studio (local, free)
  - OpenAI (cloud, paid)
  - Google Gemini (cloud, free tier + paid)
  - Anthropic Claude (cloud, paid)

- **Connection Testing**: Verifies server availability before configuration
  - Tests Ollama server connection
  - Tests LM Studio server connection
  - Validates API keys for cloud providers

- **Model Selection**: Lists available models and helps you choose
  - Shows model parameters, sizes, and capabilities
  - Filters models by vision support
  - Warns about low context windows (<4096)
  - Allows skipping vision model configuration

- **Configuration Management**: Interactive menu for ongoing changes
  - Change AI provider
  - Change text model
  - Change vision model
  - Test AI connection
  - Reset configuration
  - Exit wizard

- **First-Run Detection**: Automatically runs full setup on first use

- **Safety Features**:
  - Cost warnings for cloud providers
  - Confirmation prompts for destructive actions
  - Clear error messages

### 3. Ollama Inspector Tool

**Command**: `modulle-inspect-ollama [server_url]`

**Location**: `modulle/cli/ollama_inspector.py`

**Purpose**: Inspect Ollama models and their capabilities

**Features**:
- Lists all models on Ollama server
- Shows detailed model information:
  - Model name and family
  - Parameter count and size
  - Context window size (default and configured)
  - Vision capabilities
  - Quantization level
  - Format type
- Accepts custom server URL as argument
- Provides summary statistics
- Error handling for connection failures

**Usage**:
```bash
# Default server (localhost:11434)
modulle-inspect-ollama

# Custom server
modulle-inspect-ollama http://192.168.1.100:11434

# With Python
python -m modulle.cli.ollama_inspector
```

### 4. Enhanced config.py

**Location**: `modulle/config.py`

**New Features**:
- `_load_user_config()`: Loads `~/.modulle.json` if it exists
- `get_config(key, default)`: Get config value with priority handling
- `apply_user_config()`: Applies user config to module globals
- Automatic loading on module import

**Configuration Priority** (highest to lowest):
1. User config file (`~/.modulle.json`)
2. Environment variables (`OLLAMA_TEXT_MODEL`, etc.)
3. Defaults in `config.py`

**Supported Configuration Keys**:
- `ollama_base_url`, `ollama_text_model`, `ollama_vision_model`
- `lm_studio_base_url`, `lm_studio_text_model`, `lm_studio_vision_model`
- `openai_api_key`, `openai_text_model`, `openai_vision_model`, `openai_base_url`
- `gemini_api_key`, `gemini_text_model`, `gemini_vision_model`
- `anthropic_api_key`, `claude_text_model`, `claude_vision_model`

### 5. CLI Entry Points

**Location**: `pyproject.toml`

**Added Scripts**:
```toml
[project.scripts]
modulle-config = "modulle.cli.config_wizard:main"
modulle-inspect-ollama = "modulle.cli.ollama_inspector:main"
```

**Installation**: Automatically available after `pip install -e .`

### 6. Example Configuration File

**Location**: `.modulle.json.example`

**Features**:
- Complete example showing all providers
- Includes comments explaining each section
- Shows usage instructions
- Explains configuration priority
- Can be copied to `~/.modulle.json` and customized

### 7. Updated Documentation

**README.md Updates**:
- New "First-Time Setup" section in Quick Start
- Expanded "Configuration" section with:
  - Interactive wizard instructions
  - User config file examples
  - Environment variable examples
  - Ollama inspector usage
  - Configuration priority explanation
- Updated "Getting Started" to include `modulle-config` step

### 8. .gitignore Updates

**Added Entry**: `.modulle.json`

**Reason**: User configuration files should not be committed to version control (contains API keys and user-specific settings)

## File Structure

```
modulle/
├── cli/
│   ├── __init__.py           # CLI module exports
│   ├── config_wizard.py      # Interactive configuration wizard
│   └── ollama_inspector.py   # Ollama model inspection tool
├── config.py                 # Enhanced with config file loading
└── ...

.modulle.json.example         # Example configuration file
.gitignore                    # Updated to ignore .modulle.json
pyproject.toml                # Updated with CLI entry points
README.md                     # Updated with configuration docs
```

## Usage Examples

### First-Time Setup

```bash
# Install ModuLLe
pip install -e .

# Run configuration wizard
modulle-config

# Follow prompts to:
# 1. Select AI provider
# 2. Test connection
# 3. Choose models
# 4. Save configuration to ~/.modulle.json
```

### Using the Configuration

```python
from modulle import create_ai_client

# Automatically uses settings from ~/.modulle.json
client, text_proc, vision_proc = create_ai_client('ollama')

# Or override specific settings
client, text_proc, vision_proc = create_ai_client(
    provider='ollama',
    text_model='mistral'  # Override model from config
)
```

### Inspecting Ollama Models

```bash
# List all models and their capabilities
modulle-inspect-ollama

# Output shows:
# - Model names
# - Parameter counts and sizes
# - Context windows
# - Vision capabilities
# - Summary statistics
```

### Modifying Configuration

```bash
# Run wizard again to modify settings
modulle-config

# Choose from menu:
# [1] Change AI provider
# [2] Change text model
# [3] Change vision model
# [4] Test AI connection
# [5] Reset configuration
# [6] Exit
```

### Manual Configuration

Edit `~/.modulle.json` directly:

```json
{
  "provider": "ollama",
  "ollama_base_url": "http://localhost:11434",
  "ollama_text_model": "llama2",
  "ollama_vision_model": "llava",
  "first_run_complete": true
}
```

## Benefits

1. **User-Friendly**: No need to edit code or set environment variables
2. **Interactive**: Wizard guides users through setup with validation
3. **Persistent**: Settings saved to `~/.modulle.json` and persist across sessions
4. **Flexible**: Three configuration methods (file, env vars, code)
5. **Safe**: Cost warnings, connection testing, and error handling
6. **Discoverable**: CLI commands available via `modulle-config` and `modulle-inspect-ollama`
7. **Well-Documented**: README, example file, and inline help
8. **Clean**: No hardcoded credentials in code or repo

## Migration from Environment Variables

If you're currently using environment variables, you can:

1. **Keep using them**: Env vars still work (medium priority)
2. **Switch to config file**: Run `modulle-config` to migrate to `~/.modulle.json`
3. **Use both**: Config file overrides env vars, so you can use env vars as fallback

## Next Steps

Users can now:
1. Run `modulle-config` for first-time setup
2. Edit `~/.modulle.json` for quick changes
3. Use `modulle-inspect-ollama` to discover available models
4. Switch providers easily by changing one line in `~/.modulle.json`

No more hardcoded configuration!
