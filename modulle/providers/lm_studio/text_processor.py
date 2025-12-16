"""
Generic text processing with LM Studio.

Provides a thin wrapper around LM Studio's OpenAI-compatible API.
All application logic should be built on top of this generic interface.
"""
from typing import Optional, List, Dict, Any
from .client import LMStudioClient
from modulle.utils.logging_config import get_logger
from modulle.config import DEFAULT_TEMPERATURE
from modulle.base import BaseTextProcessor

logger = get_logger(__name__.replace("modulle.providers.", ""))


class LMStudioTextProcessor(BaseTextProcessor):
    """
    Generic text processor using LM Studio.

    Implements the BaseTextProcessor interface for LM Studio models.
    Applications build domain-specific logic by crafting prompts.
    """

    def __init__(self, model: str, base_url: Optional[str] = None):
        """
        Initialize LM Studio text processor.

        Args:
            model: LM Studio model name to use
            base_url: LM Studio server URL (optional, uses default if not provided)
        """
        self.model = model
        self.client = LMStudioClient(base_url=base_url) if base_url else LMStudioClient()
        logger.info(f"LM Studio text processor initialized with model: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate text from a prompt using LM Studio.

        This is the primary method for all text generation tasks.
        Applications build their logic by crafting appropriate prompts.

        Args:
            prompt: User prompt/instruction
            system_prompt: System prompt to set model behavior (optional)
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative)
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
                logger.warning("Empty response from LM Studio generate")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in LM Studio text generation: {e}")
            return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Multi-turn chat interaction using LM Studio.

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
                logger.warning("Empty response from LM Studio chat")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in LM Studio chat: {e}")
            return None


# Backward compatibility alias
LMStudioTextClient = LMStudioTextProcessor
