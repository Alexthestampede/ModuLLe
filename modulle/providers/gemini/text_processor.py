"""
Generic text processing with Google Gemini.

Provides a thin wrapper around Gemini's text generation API.
All application logic should be built on top of this generic interface.
"""
from typing import Optional, List, Dict, Any
from .client import GeminiClient
from modulle.utils.logging_config import get_logger
from modulle.config import DEFAULT_TEMPERATURE
from modulle.base import BaseTextProcessor

logger = get_logger(__name__.replace("modulle.providers.", ""))


class GeminiTextProcessor(BaseTextProcessor):
    """
    Generic text processor using Google Gemini.

    Implements the BaseTextProcessor interface for Gemini models.
    Applications build domain-specific logic by crafting prompts.
    """

    def __init__(self, model: str, api_key: str):
        """
        Initialize Gemini text processor.

        Args:
            model: Gemini model name (gemini-1.5-flash, gemini-1.5-pro, etc.)
            api_key: Google API key
        """
        self.model = model
        self.client = GeminiClient(api_key=api_key)
        logger.info(f"Gemini text processor initialized with model: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate text from a prompt using Gemini.

        This is the primary method for all text generation tasks.
        Applications build their logic by crafting appropriate prompts.

        Args:
            prompt: User prompt/instruction
            system_prompt: System prompt to set model behavior (optional)
            temperature: Sampling temperature (0.0=deterministic, 2.0=very creative)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Generated text string, or None on error
        """
        try:
            response = self.client.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=temperature,
                model=self.model
            )

            if not response:
                logger.warning("Empty response from Gemini generate")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in Gemini text generation: {e}")
            return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Multi-turn chat interaction using Gemini.

        Args:
            messages: Conversation history as list of message dicts
            temperature: Sampling temperature (0.0=deterministic, 2.0=very creative)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Generated response string, or None on error
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=temperature
            )

            if not response:
                logger.warning("Empty response from Gemini chat")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in Gemini chat: {e}")
            return None


# Backward compatibility alias
GeminiTextClient = GeminiTextProcessor
