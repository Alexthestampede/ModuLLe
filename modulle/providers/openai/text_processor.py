"""
Generic text processing with OpenAI.

Provides a thin wrapper around OpenAI's text generation API.
All application logic should be built on top of this generic interface.
"""
from typing import Optional, List, Dict, Any
from .client import OpenAIClient
from modulle.utils.logging_config import get_logger
from modulle.config import DEFAULT_TEMPERATURE
from modulle.base import BaseTextProcessor

logger = get_logger(__name__.replace("modulle.providers.", ""))


class OpenAITextProcessor(BaseTextProcessor):
    """
    Generic text processor using OpenAI.

    Implements the BaseTextProcessor interface for OpenAI models.
    Applications build domain-specific logic by crafting prompts.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize OpenAI text processor.

        Args:
            model: OpenAI model name (gpt-4o-mini, gpt-3.5-turbo, gpt-4o, etc.)
            api_key: OpenAI API key (optional, will use env var if not provided)

        Raises:
            ValueError: If API key is not found
        """
        self.model = model
        try:
            self.client = OpenAIClient(api_key=api_key)
            logger.info(f"OpenAI text processor initialized with model: {model}")
        except ValueError as e:
            logger.error(f"Failed to initialize OpenAI text processor: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate text from a prompt using OpenAI.

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
                model=self.model,
                prompt=prompt,
                system=system_prompt,
                temperature=temperature
            )

            if not response:
                logger.warning("Empty response from OpenAI generate")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in OpenAI text generation: {e}")
            return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Multi-turn chat interaction using OpenAI.

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
                logger.warning("Empty response from OpenAI chat")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in OpenAI chat: {e}")
            return None


# Backward compatibility alias
OpenAITextClient = OpenAITextProcessor
