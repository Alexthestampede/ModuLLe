"""
OpenAI API client for RSS Feed Processor

This module provides integration with OpenAI's API endpoints for text generation
and vision capabilities using GPT-4 and GPT-3.5-turbo models.
"""
import os
import requests
from typing import Optional, List, Dict, Any
from modulle.utils.logging_config import get_logger
from modulle.config import REQUEST_TIMEOUT
from modulle.base import BaseAIClient

logger = get_logger(__name__.replace("modulle.providers.", ""))


class OpenAIClient(BaseAIClient):
    """
    Client for interacting with OpenAI API.

    Implements the BaseAIClient interface for consistent API across different
    AI providers.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1"):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: OpenAI API base URL

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        # Get API key from parameter, environment, or config
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

        if not self.api_key:
            error_msg = (
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or provide api_key parameter. Get your API key from: https://platform.openai.com/api-keys"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        logger.info("OpenAI client initialized successfully")

    def health_check(self) -> bool:
        """
        Check if OpenAI API is available and API key is valid.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Try to list models as a health check
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=5
            )

            if response.status_code == 401:
                logger.error("OpenAI API authentication failed - invalid API key")
                return False
            elif response.status_code == 429:
                logger.warning("OpenAI API rate limit exceeded")
                return True  # API is available, just rate limited

            response.raise_for_status()
            logger.info("OpenAI API is available")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API health check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        """
        List available models from OpenAI API.

        Returns:
            List of model IDs, or empty list on error
        """
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            data = response.json()
            models = [model['id'] for model in data.get('data', [])]

            logger.info(f"Retrieved {len(models)} models from OpenAI")
            logger.debug(f"Available models: {models[:5]}...")  # Log first 5

            return models

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list OpenAI models: {e}")
            return []

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        images: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Generate text using OpenAI model.

        This method converts the simple prompt interface to OpenAI's chat format.

        Args:
            model: Model ID to use (e.g., 'gpt-4o-mini', 'gpt-4o')
            prompt: User prompt
            system: System prompt (optional)
            temperature: Generation temperature (0.0 to 1.0)
            images: List of base64-encoded images for vision models (optional)

        Returns:
            Generated text, or None on error
        """
        # Build messages array
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        # Handle vision models with images
        if images:
            content = [{"type": "text", "text": prompt}]
            for image_data in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        return self.chat(model, messages, temperature)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.3
    ) -> Optional[str]:
        """
        Chat using OpenAI's chat completions API.

        Args:
            model: Model ID to use
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Generation temperature (0.0 to 1.0)

        Returns:
            Generated response, or None on error
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1000  # Reasonable default for summaries
            }

            logger.debug(f"Sending chat request to OpenAI model: {model}")

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=REQUEST_TIMEOUT * 3  # Longer timeout for generation
            )

            # Handle rate limiting
            if response.status_code == 429:
                logger.error("OpenAI API rate limit exceeded")
                return None

            # Handle authentication errors
            if response.status_code == 401:
                logger.error("OpenAI API authentication failed - check API key")
                return None

            response.raise_for_status()

            data = response.json()

            # Extract content from response
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content'].strip()
                logger.debug(f"Generated {len(content)} characters")
                return content
            else:
                logger.error("No choices in OpenAI response")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"OpenAI request timeout after {REQUEST_TIMEOUT * 3} seconds")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI chat request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI chat: {e}")
            return None
