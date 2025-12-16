"""
Configuration constants for ModuLLe AI provider abstraction.

These are generic defaults that can be overridden when creating AI clients.
No application-specific logic here - just provider configurations.

Configuration priority (highest to lowest):
1. User config file (~/.modulle.json)
2. Environment variables
3. Defaults in this file
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Default generation parameters
DEFAULT_TEMPERATURE = 0.7  # Balanced between deterministic and creative
DEFAULT_MAX_TOKENS = None  # Let the provider decide

# HTTP request settings (for cloud APIs and fetching resources)
USER_AGENT = 'ModuLLe/0.2.0 (AI Provider Abstraction)'
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
RETRY_BACKOFF = 2  # multiplier for exponential backoff

# Logging settings
LOG_LEVEL = os.getenv('MODULLE_LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================================
# Provider-specific configurations
# ============================================================================

# Ollama (local open-source models)
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_TEXT_MODEL = os.getenv('OLLAMA_TEXT_MODEL', 'llama2')
OLLAMA_VISION_MODEL = os.getenv('OLLAMA_VISION_MODEL', 'llava')

# LM Studio (local models with OpenAI-compatible API)
LM_STUDIO_BASE_URL = os.getenv('LM_STUDIO_BASE_URL', 'http://localhost:1234')
LM_STUDIO_TEXT_MODEL = os.getenv('LM_STUDIO_TEXT_MODEL', 'local-model')
LM_STUDIO_VISION_MODEL = os.getenv('LM_STUDIO_VISION_MODEL', 'local-model')

# OpenAI (cloud API)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_TEXT_MODEL = os.getenv('OPENAI_TEXT_MODEL', 'gpt-4o-mini')
OPENAI_VISION_MODEL = os.getenv('OPENAI_VISION_MODEL', 'gpt-4o')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

# Google Gemini (cloud API)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', os.getenv('GOOGLE_API_KEY', ''))
GEMINI_TEXT_MODEL = os.getenv('GEMINI_TEXT_MODEL', 'gemini-1.5-flash')
GEMINI_VISION_MODEL = os.getenv('GEMINI_VISION_MODEL', 'gemini-1.5-flash')

# Anthropic Claude (cloud API)
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
CLAUDE_TEXT_MODEL = os.getenv('CLAUDE_TEXT_MODEL', 'claude-3-5-haiku-20241022')
CLAUDE_VISION_MODEL = os.getenv('CLAUDE_VISION_MODEL', 'claude-3-5-sonnet-20241022')


# ============================================================================
# User Configuration File Support
# ============================================================================

def _get_config_file_path() -> Path:
    """Get the path to the user configuration file."""
    return Path.home() / '.modulle.json'


def _load_user_config() -> Optional[Dict[str, Any]]:
    """
    Load user configuration from ~/.modulle.json if it exists.

    Returns:
        Configuration dictionary or None if file doesn't exist or is invalid
    """
    config_file = _get_config_file_path()

    if not config_file.exists():
        return None

    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load user config from {config_file}: {e}")
        return None


def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value with priority: user config > env var > default.

    Args:
        key: Configuration key to retrieve
        default: Default value if not found anywhere

    Returns:
        Configuration value from highest priority source
    """
    # Try user config file first
    user_config = _load_user_config()
    if user_config and key in user_config:
        return user_config[key]

    # Fall back to environment variable (if it exists)
    env_key = key.upper()
    env_value = os.getenv(env_key)
    if env_value is not None:
        return env_value

    # Fall back to default
    return default


def apply_user_config():
    """
    Apply user configuration from ~/.modulle.json to override defaults.
    This function updates the module-level constants with user config values.

    Call this at module initialization or when you want to reload config.
    """
    user_config = _load_user_config()

    if not user_config:
        return

    # Map of config keys to module globals
    config_mapping = {
        # Ollama
        'ollama_base_url': 'OLLAMA_BASE_URL',
        'ollama_text_model': 'OLLAMA_TEXT_MODEL',
        'ollama_vision_model': 'OLLAMA_VISION_MODEL',

        # LM Studio
        'lm_studio_base_url': 'LM_STUDIO_BASE_URL',
        'lm_studio_text_model': 'LM_STUDIO_TEXT_MODEL',
        'lm_studio_vision_model': 'LM_STUDIO_VISION_MODEL',

        # OpenAI
        'openai_api_key': 'OPENAI_API_KEY',
        'openai_text_model': 'OPENAI_TEXT_MODEL',
        'openai_vision_model': 'OPENAI_VISION_MODEL',
        'openai_base_url': 'OPENAI_BASE_URL',

        # Gemini
        'gemini_api_key': 'GEMINI_API_KEY',
        'gemini_text_model': 'GEMINI_TEXT_MODEL',
        'gemini_vision_model': 'GEMINI_VISION_MODEL',

        # Claude
        'anthropic_api_key': 'ANTHROPIC_API_KEY',
        'claude_text_model': 'CLAUDE_TEXT_MODEL',
        'claude_vision_model': 'CLAUDE_VISION_MODEL',
    }

    # Update globals with user config values
    globals_dict = globals()
    for config_key, global_name in config_mapping.items():
        if config_key in user_config:
            globals_dict[global_name] = user_config[config_key]


# Apply user config on module import
apply_user_config()
