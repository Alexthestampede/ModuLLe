"""
Generic vision processing with LM Studio.

Provides image analysis capabilities using LM Studio vision models.
All application logic should be built on top of this generic interface.

Note: LM Studio vision support depends on the loaded model having vision capabilities.
"""
from typing import Optional
from .client import LMStudioClient
from modulle.utils.logging_config import get_logger
from modulle.config import DEFAULT_TEMPERATURE
from modulle.base import BaseVisionProcessor

logger = get_logger(__name__.replace("modulle.providers.", ""))


class LMStudioVisionProcessor(BaseVisionProcessor):
    """
    Generic vision processor using LM Studio.

    Implements the BaseVisionProcessor interface for LM Studio vision models.
    Applications build domain-specific image processing by crafting prompts.

    Note: Requires a vision-capable model to be loaded in LM Studio.
    """

    def __init__(self, model: str, base_url: Optional[str] = None):
        """
        Initialize LM Studio vision processor.

        Args:
            model: LM Studio model name (must be a vision-capable model)
            base_url: LM Studio server URL (optional, uses default if not provided)
        """
        self.model = model
        self.client = LMStudioClient(base_url=base_url) if base_url else LMStudioClient()
        logger.info(f"LM Studio vision processor initialized with model: {model}")

    def analyze_image(
        self,
        image_data: str,
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Analyze an image with a custom prompt using LM Studio.

        This is the primary method for all vision tasks. Applications
        build their logic by crafting appropriate prompts.

        Args:
            image_data: Base64-encoded image data
            prompt: Instruction describing what to analyze
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Analysis result as text string, or None on error

        Note:
            The model loaded in LM Studio must support vision inputs.
        """
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                images=[image_data]
            )

            if not response:
                logger.warning("Empty response from LM Studio vision analysis")
                return None

            return response

        except Exception as e:
            logger.error(f"Error in LM Studio vision analysis: {e}")
            return None


# Backward compatibility alias
LMStudioVisionClient = LMStudioVisionProcessor
