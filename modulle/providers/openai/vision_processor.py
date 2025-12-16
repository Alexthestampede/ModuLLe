"""
Generic vision processing with OpenAI.

Provides image analysis capabilities using OpenAI vision models.
All application logic should be built on top of this generic interface.
"""
from typing import Optional
from .client import OpenAIClient
from modulle.utils.logging_config import get_logger
from modulle.config import DEFAULT_TEMPERATURE
from modulle.base import BaseVisionProcessor

logger = get_logger(__name__.replace("modulle.providers.", ""))


class OpenAIVisionProcessor(BaseVisionProcessor):
    """
    Generic vision processor using OpenAI.

    Implements the BaseVisionProcessor interface for OpenAI vision models.
    Applications build domain-specific image processing by crafting prompts.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize OpenAI vision processor.

        Args:
            model: OpenAI vision model (gpt-4o, gpt-4-turbo, etc.)
            api_key: OpenAI API key (optional, will use env var if not provided)

        Raises:
            ValueError: If API key is not found

        Note:
            gpt-4o is recommended as it has native vision capabilities
        """
        self.model = model
        try:
            self.client = OpenAIClient(api_key=api_key)
            logger.info(f"OpenAI vision processor initialized with model: {model}")
        except ValueError as e:
            logger.error(f"Failed to initialize OpenAI vision processor: {e}")
            raise

    def analyze_image(
        self,
        image_data: str,
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Analyze an image with a custom prompt using OpenAI.

        This is the primary method for all vision tasks. Applications
        build their logic by crafting appropriate prompts.

        Args:
            image_data: Base64-encoded image data
            prompt: Instruction describing what to analyze
            temperature: Sampling temperature (0.0=deterministic, 2.0=very creative)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Analysis result as text string, or None on error
        """
        try:
            # OpenAI expects images in the chat format
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                images=[image_data]
            )

            if not response:
                logger.warning("Empty response from OpenAI vision analysis")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in OpenAI vision analysis: {e}")
            return None
