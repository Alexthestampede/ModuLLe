"""
Factory for creating AI client and text processor instances.

This module provides factory functions to create the appropriate AI client
and text processor based on the configured provider (Ollama, LM Studio, OpenAI, Gemini, Claude).
"""
from typing import Tuple, Optional
from .utils.logging_config import get_logger
from .base import BaseAIClient, BaseTextProcessor
from . import config

logger = get_logger(__name__)


def create_ai_client(
    provider: str = 'ollama',
    text_model: Optional[str] = None,
    vision_model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Tuple[BaseAIClient, BaseTextProcessor, Optional[object]]:
    """
    Create AI client, text processor, and vision processor instances based on specified provider.

    Args:
        provider: AI provider name ('ollama', 'lm_studio', 'openai', 'gemini', 'claude')
        text_model: Model name for text processing (optional, uses config default if not provided)
        vision_model: Model name for vision processing (optional, uses config default if not provided)
        base_url: Base URL for local providers (Ollama, LM Studio)
        api_key: API key for cloud providers (OpenAI, Gemini, Claude)
        **kwargs: Additional provider-specific arguments

    Returns:
        Tuple of (BaseAIClient, BaseTextProcessor, VisionProcessor) instances
        VisionProcessor may be None if not available for the provider

    Raises:
        ValueError: If configured provider is unknown or API key is missing
        ImportError: If provider module is not available

    Example:
        # Local provider (Ollama)
        client, text_proc, vision_proc = create_ai_client(
            provider='ollama',
            text_model='llama2',
            base_url='http://localhost:11434'
        )

        # Cloud provider (OpenAI)
        client, text_proc, vision_proc = create_ai_client(
            provider='openai',
            text_model='gpt-4o-mini',
            api_key='your-api-key'
        )
    """
    provider = provider.lower()
    logger.info(f"Creating AI client for provider: {provider}")

    if provider == 'ollama':
        from .providers.ollama.client import OllamaClient
        from .providers.ollama.text_processor import OllamaTextClient
        from .providers.ollama.vision_processor import OllamaVisionClient

        # Get configuration with defaults
        base_url = base_url or config.OLLAMA_BASE_URL
        text_model = text_model or config.OLLAMA_TEXT_MODEL
        vision_model = vision_model or config.OLLAMA_VISION_MODEL

        logger.info(f"Ollama config - URL: {base_url}, Text: {text_model}, Vision: {vision_model}")

        client = OllamaClient(base_url=base_url)
        text_processor = OllamaTextClient(model=text_model, base_url=base_url)
        vision_processor = OllamaVisionClient(model=vision_model, base_url=base_url) if vision_model else None

        logger.info("Ollama client initialized successfully")
        return client, text_processor, vision_processor

    elif provider == 'lm_studio' or provider == 'lmstudio':
        from .providers.lm_studio.client import LMStudioClient
        from .providers.lm_studio.text_processor import LMStudioTextClient
        from .providers.lm_studio.vision_processor import LMStudioVisionClient

        # Get configuration with defaults
        base_url = base_url or config.LM_STUDIO_BASE_URL
        text_model = text_model or config.LM_STUDIO_TEXT_MODEL

        logger.info(f"LM Studio config - URL: {base_url}, Model: {text_model}")

        client = LMStudioClient(base_url=base_url)
        text_processor = LMStudioTextClient(model=text_model, base_url=base_url)
        vision_processor = LMStudioVisionClient(model=text_model, base_url=base_url)

        logger.info("LM Studio client initialized successfully")
        return client, text_processor, vision_processor

    elif provider == 'openai':
        from .providers.openai.client import OpenAIClient
        from .providers.openai.text_processor import OpenAITextProcessor
        from .providers.openai.vision_processor import OpenAIVisionProcessor

        # Get configuration with defaults
        api_key = api_key or config.OPENAI_API_KEY
        text_model = text_model or config.OPENAI_TEXT_MODEL
        vision_model = vision_model or config.OPENAI_VISION_MODEL

        if not api_key:
            error_msg = (
                "OpenAI provider requires API key. Please provide api_key parameter or set "
                "OPENAI_API_KEY environment variable. Get your API key from: "
                "https://platform.openai.com/api-keys"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"OpenAI config - Text: {text_model}, Vision: {vision_model}")

        client = OpenAIClient(api_key=api_key)
        text_processor = OpenAITextProcessor(model=text_model, api_key=api_key)
        vision_processor = OpenAIVisionProcessor(model=vision_model, api_key=api_key)

        logger.info("OpenAI client initialized successfully")
        return client, text_processor, vision_processor

    elif provider == 'gemini':
        from .providers.gemini.client import GeminiClient
        from .providers.gemini.text_processor import GeminiTextClient
        from .providers.gemini.vision_processor import GeminiVisionClient

        # Get configuration with defaults
        api_key = api_key or config.GEMINI_API_KEY
        text_model = text_model or config.GEMINI_TEXT_MODEL
        vision_model = vision_model or config.GEMINI_VISION_MODEL

        if not api_key:
            error_msg = (
                "Gemini provider requires API key. Please provide api_key parameter or set "
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable. Get your API key from: "
                "https://aistudio.google.com/app/apikey"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Gemini config - Text: {text_model}, Vision: {vision_model}")

        client = GeminiClient(api_key=api_key)
        text_processor = GeminiTextClient(api_key=api_key, model=text_model)
        vision_processor = GeminiVisionClient(api_key=api_key, model=vision_model)

        logger.info("Gemini client initialized successfully")
        return client, text_processor, vision_processor

    elif provider == 'claude' or provider == 'anthropic':
        from .providers.claude.client import ClaudeClient
        from .providers.claude.text_processor import ClaudeTextClient
        from .providers.claude.vision_processor import ClaudeVisionClient

        # Get configuration with defaults
        api_key = api_key or config.ANTHROPIC_API_KEY
        text_model = text_model or config.CLAUDE_TEXT_MODEL
        vision_model = vision_model or config.CLAUDE_VISION_MODEL

        if not api_key:
            error_msg = (
                "Claude provider requires API key. Please provide api_key parameter or set "
                "ANTHROPIC_API_KEY environment variable. Get your API key from: "
                "https://console.anthropic.com/"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Claude config - Text: {text_model}, Vision: {vision_model}")

        client = ClaudeClient(api_key=api_key)
        text_processor = ClaudeTextClient(api_key=api_key, model=text_model)
        vision_processor = ClaudeVisionClient(api_key=api_key, model=vision_model)

        logger.info("Claude client initialized successfully")
        return client, text_processor, vision_processor

    else:
        error_msg = (
            f"Unknown AI provider: {provider}. "
            f"Supported providers: 'ollama', 'lm_studio', 'openai', 'gemini', 'claude'"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
