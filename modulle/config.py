"""
Configuration constants for ModuLLe AI provider abstraction.

These are generic defaults that can be overridden when creating AI clients.
No application-specific logic here - just provider configurations.
"""
import os

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
