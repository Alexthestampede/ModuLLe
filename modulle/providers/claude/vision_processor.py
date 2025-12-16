"""
Generic vision processing with Anthropic Claude.

Provides image analysis capabilities using Claude vision models.
All application logic should be built on top of this generic interface.
"""
from typing import Optional
from .client import ClaudeClient
from modulle.utils.logging_config import get_logger
from modulle.config import DEFAULT_TEMPERATURE
from modulle.base import BaseVisionProcessor

logger = get_logger(__name__.replace("modulle.providers.", ""))


class ClaudeVisionProcessor(BaseVisionProcessor):
    """
    Generic vision processor using Anthropic Claude.

    Implements the BaseVisionProcessor interface for Claude vision models.
    Applications build domain-specific image processing by crafting prompts.
    """

    def __init__(self, model: str, api_key: str):
        """
        Initialize Claude vision processor.

        Args:
            model: Claude vision model (claude-3-5-sonnet-20241022, claude-3-opus, etc.)
            api_key: Anthropic API key

        Note:
            Claude 3.5 Sonnet and Claude 3 Opus have strong vision capabilities
        """
        self.model = model
        self.client = ClaudeClient(api_key=api_key)
        logger.info(f"Claude vision processor initialized with model: {model}")

    def analyze_image(
        self,
        image_data: str,
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Analyze an image with a custom prompt using Claude.

        This is the primary method for all vision tasks. Applications
        build their logic by crafting appropriate prompts.

        Args:
            image_data: Base64-encoded image data
            prompt: Instruction describing what to analyze
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Analysis result as text string, or None on error
        """
        try:
            # Claude supports vision through the messages API
            response = self.client.analyze_image(
                image_data=image_data,
                prompt=prompt,
                temperature=temperature,
                model=self.model
            )

            if not response:
                logger.warning("Empty response from Claude vision analysis")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in Claude vision analysis: {e}")
            return None


# Backward compatibility alias
ClaudeVisionClient = ClaudeVisionProcessor
