"""
Generic vision processing with Ollama.

Provides image analysis capabilities using Ollama vision models.
All application logic should be built on top of this generic interface.
"""
from typing import Optional
from .client import OllamaClient
from modulle.utils.logging_config import get_logger
from modulle.config import DEFAULT_TEMPERATURE
from modulle.base import BaseVisionProcessor

logger = get_logger(__name__.replace("modulle.providers.", ""))


class OllamaVisionProcessor(BaseVisionProcessor):
    """
    Generic vision processor using Ollama.

    Implements the BaseVisionProcessor interface for Ollama vision models.
    Applications build domain-specific image processing by crafting prompts.
    """

    def __init__(self, model: str, base_url: Optional[str] = None):
        """
        Initialize Ollama vision processor.

        Args:
            model: Ollama vision model name (llava, bakllava, etc.)
            base_url: Ollama server URL (optional, uses default if not provided)
        """
        self.model = model
        self.client = OllamaClient(base_url=base_url) if base_url else OllamaClient()
        logger.info(f"Ollama vision processor initialized with model: {model}")

    def analyze_image(
        self,
        image_data: str,
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Analyze an image with a custom prompt using Ollama.

        This is the primary method for all vision tasks. Applications
        build their logic by crafting appropriate prompts.

        Examples:
            # Image description
            result = processor.analyze_image(
                image_data=base64_image,
                prompt="Describe this image in detail"
            )

            # OCR / Text extraction
            result = processor.analyze_image(
                image_data=base64_image,
                prompt="Extract all text from this image"
            )

            # Object detection
            result = processor.analyze_image(
                image_data=base64_image,
                prompt="List all objects visible in this image"
            )

        Args:
            image_data: Base64-encoded image data
            prompt: Instruction describing what to analyze
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Analysis result as text string, or None on error
        """
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                images=[image_data]
            )

            if not response:
                logger.warning("Empty response from Ollama vision analysis")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in Ollama vision analysis: {e}")
            return None


# Backward compatibility alias
OllamaVisionClient = OllamaVisionProcessor
