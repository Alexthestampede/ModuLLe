"""
Generic vision processing with Google Gemini.

Provides image analysis capabilities using Gemini vision models.
All application logic should be built on top of this generic interface.
"""
from typing import Optional
from .client import GeminiClient
from modulle.utils.logging_config import get_logger
from modulle.config import DEFAULT_TEMPERATURE
from modulle.base import BaseVisionProcessor

logger = get_logger(__name__.replace("modulle.providers.", ""))


class GeminiVisionProcessor(BaseVisionProcessor):
    """
    Generic vision processor using Google Gemini.

    Implements the BaseVisionProcessor interface for Gemini vision models.
    Applications build domain-specific image processing by crafting prompts.
    """

    def __init__(self, model: str, api_key: str):
        """
        Initialize Gemini vision processor.

        Args:
            model: Gemini vision model (gemini-1.5-flash, gemini-1.5-pro, etc.)
            api_key: Google API key

        Note:
            Gemini 1.5 models have native multimodal capabilities
        """
        self.model = model
        self.client = GeminiClient(api_key=api_key)
        logger.info(f"Gemini vision processor initialized with model: {model}")

    def analyze_image(
        self,
        image_data: str,
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Analyze an image with a custom prompt using Gemini.

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
            # Gemini supports multimodal inputs natively
            response = self.client.analyze_image(
                image_data=image_data,
                prompt=prompt,
                temperature=temperature,
                model=self.model
            )

            if not response:
                logger.warning("Empty response from Gemini vision analysis")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in Gemini vision analysis: {e}")
            return None


# Backward compatibility alias
GeminiVisionClient = GeminiVisionProcessor
