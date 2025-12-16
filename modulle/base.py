"""
Base abstract classes for AI provider abstraction layer.

ModuLLe provides a unified, generic interface for AI/LLM providers.
This module defines the core abstractions that all providers must implement.

The design philosophy is:
- Generic: No application-specific logic (no article processing, RSS, etc.)
- Simple: Minimal interface with essential operations only
- Flexible: Applications build their logic on top of these primitives
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseAIClient(ABC):
    """
    Abstract base class for AI provider clients.

    Provides low-level access to AI models with basic operations:
    - Health checks and model discovery
    - Raw text generation
    - Chat-based interactions

    All AI providers (Ollama, LM Studio, OpenAI, etc.) must implement
    this interface to ensure consistent API across providers.
    """

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the AI server/API is available and responsive.

        Returns:
            True if server is available, False otherwise
        """
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """
        List available models on the AI server/API.

        Returns:
            List of model names, or empty list on error
        """
        pass

    @abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        images: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Generate text using the AI model (single-turn generation).

        This is the most basic text generation primitive. Use this for:
        - Simple completion tasks
        - Single-turn Q&A
        - Text transformation (summarization, translation, etc.)

        Args:
            model: Model name to use
            prompt: User prompt/instruction
            system: System prompt to set behavior (optional)
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum tokens to generate (optional, provider-specific)
            images: Base64-encoded images for vision models (optional)

        Returns:
            Generated text string, or None on error
        """
        pass

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Chat using the AI model (multi-turn conversation).

        Use this for conversational interactions where context matters.

        Args:
            model: Model name to use
            messages: List of message dicts with 'role' and 'content' keys
                     Example: [
                         {"role": "system", "content": "You are helpful"},
                         {"role": "user", "content": "Hello"},
                         {"role": "assistant", "content": "Hi there!"},
                         {"role": "user", "content": "How are you?"}
                     ]
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum tokens to generate (optional, provider-specific)

        Returns:
            Generated response string, or None on error
        """
        pass


class BaseTextProcessor(ABC):
    """
    Abstract base class for high-level text processing operations.

    Provides convenience methods built on top of BaseAIClient.generate()
    and BaseAIClient.chat() for common text processing tasks.

    This is a generic text processing interface - applications can extend
    this or use the methods directly to build domain-specific functionality.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate text from a prompt (generic text generation).

        This is the primary method for all text generation tasks.
        Applications build their logic by crafting appropriate prompts.

        Examples:
            # Summarization
            result = processor.generate(
                prompt=f"Summarize this text: {text}",
                system_prompt="You are a professional summarizer",
                temperature=0.3
            )

            # Translation
            result = processor.generate(
                prompt=f"Translate to French: {text}",
                temperature=0.2
            )

            # Classification
            result = processor.generate(
                prompt=f"Is this spam? Answer yes/no: {text}",
                temperature=0.1
            )

        Args:
            prompt: User prompt/instruction
            system_prompt: System prompt to set model behavior (optional)
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Generated text string, or None on error
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Multi-turn chat interaction (generic conversation).

        Use this when you need to maintain conversation context
        across multiple turns.

        Args:
            messages: Conversation history as list of message dicts
                     Each dict has 'role' (system/user/assistant) and 'content'
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Generated response string, or None on error
        """
        pass


class BaseVisionProcessor(ABC):
    """
    Abstract base class for vision/image processing operations.

    Provides generic image analysis capabilities. Applications build
    domain-specific image processing by crafting appropriate prompts.
    """

    @abstractmethod
    def analyze_image(
        self,
        image_data: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Analyze an image with a custom prompt (generic image analysis).

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

            # Classification
            result = processor.analyze_image(
                image_data=base64_image,
                prompt="Is this image safe for work? Answer yes or no"
            )

        Args:
            image_data: Base64-encoded image data
            prompt: Instruction describing what to analyze
            temperature: Sampling temperature (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Analysis result as text string, or None on error
        """
        pass
