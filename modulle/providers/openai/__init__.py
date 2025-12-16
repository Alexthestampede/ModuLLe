"""
OpenAI provider for RSS Feed Processor

This module provides OpenAI API integration for text summarization and
vision processing using GPT-4 and GPT-3.5-turbo models.

Usage:
    from src.openai_provider import OpenAIClient, OpenAITextProcessor, OpenAIVisionProcessor

    # Initialize with API key from environment
    client = OpenAIClient()
    text_processor = OpenAITextProcessor(model="gpt-4o-mini")
    vision_processor = OpenAIVisionProcessor(model="gpt-4o")

    # Or provide API key explicitly
    client = OpenAIClient(api_key="sk-...")
    text_processor = OpenAITextProcessor(api_key="sk-...")
"""

from .client import OpenAIClient
from .text_processor import OpenAITextProcessor
from .vision_processor import OpenAIVisionProcessor

__all__ = [
    'OpenAIClient',
    'OpenAITextProcessor',
    'OpenAIVisionProcessor'
]
