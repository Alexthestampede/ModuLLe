"""
ModuLLe - Modular LLM Provider Abstraction Layer

A unified interface for multiple AI/LLM providers including:
- Ollama (local)
- LM Studio (local)
- OpenAI (cloud)
- Google Gemini (cloud)
- Anthropic Claude (cloud)

Example usage:
    from modulle import create_ai_client

    client, text_processor, vision_processor = create_ai_client(
        provider='ollama',
        text_model='llama2',
        base_url='http://localhost:11434'
    )

    result = text_processor.generate_summary(
        text="Your article text here",
        title="Article Title"
    )
"""

__version__ = "0.1.0"
__author__ = "Extracted from DailyFeedSanity"

from .base import BaseAIClient, BaseTextProcessor
from .factory import create_ai_client

__all__ = [
    'BaseAIClient',
    'BaseTextProcessor',
    'create_ai_client',
]
