"""
Generic text processing with Ollama.

Provides a thin wrapper around Ollama's text generation API.
All application logic should be built on top of this generic interface.
"""
from typing import Optional, List, Dict, Any
from .client import OllamaClient
from modulle.utils.logging_config import get_logger
from modulle.config import DEFAULT_TEMPERATURE
from modulle.base import BaseTextProcessor

logger = get_logger(__name__.replace("modulle.providers.", ""))


class OllamaTextProcessor(BaseTextProcessor):
    """
    Generic text processor using Ollama.

    Implements the BaseTextProcessor interface for Ollama models.
    Applications build domain-specific logic by crafting prompts.
    """

    def __init__(self, model: str, base_url: Optional[str] = None):
        """
        Initialize Ollama text processor.

        Args:
            model: Ollama model name to use
            base_url: Ollama server URL (optional, uses default if not provided)
        """
        self.model = model
        self.client = OllamaClient(base_url=base_url) if base_url else OllamaClient()
        logger.info(f"Ollama text processor initialized with model: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate text from a prompt using Ollama.

        This is the primary method for all text generation tasks.
        Applications build their logic by crafting appropriate prompts.

        Args:
            prompt: User prompt/instruction
            system_prompt: System prompt to set model behavior (optional)
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum tokens to generate (optional, not widely supported by Ollama)

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
                logger.warning("Empty response from Ollama generate")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in Ollama text generation: {e}")
            return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Multi-turn chat interaction using Ollama.

        Args:
            messages: Conversation history as list of message dicts
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative)
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
                logger.warning("Empty response from Ollama chat")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in Ollama chat: {e}")
            return None


# Backward compatibility alias
OllamaTextClient = OllamaTextProcessor
