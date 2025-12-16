"""
Anthropic Claude API client for RSS Feed Processor

This module provides integration with Anthropic's Claude API using the
Messages API format.
"""
import requests
from typing import Optional, List, Dict, Any
from modulle.utils.logging_config import get_logger
from modulle.config import REQUEST_TIMEOUT
from modulle.base import BaseAIClient

logger = get_logger(__name__.replace("modulle.providers.", ""))


class ClaudeClient(BaseAIClient):
    """
    Client for interacting with Anthropic Claude API.

    Claude uses the Messages API format with system prompts separate from messages.
    """

    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com/v1"):
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key
            base_url: Claude API base URL
        """
        if not api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")

        self.api_key = api_key
        self.base_url = base_url.rstrip('/')

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json"
        }

    def health_check(self) -> bool:
        """
        Check if Claude API is available.

        Returns:
            True if API is available, False otherwise
        """
        try:
            # Try a minimal request to check API availability
            url = f"{self.base_url}/messages"
            headers = self._get_headers()

            payload = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hi"}]
            }

            response = requests.post(url, headers=headers, json=payload, timeout=5)

            # 200 or 400 means API is accessible (400 just means invalid request format)
            if response.status_code in [200, 400]:
                logger.info("Claude API is available")
                return True

            logger.error(f"Claude API health check failed: {response.status_code}")
            return False

        except Exception as e:
            logger.error(f"Claude API health check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        """
        List available Claude models.

        Note: Claude API doesn't have a models endpoint, so we return known models.

        Returns:
            List of model names
        """
        models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
        logger.info(f"Known Claude models: {len(models)}")
        return models

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.3, model: Optional[str] = None,
                 max_tokens: int = 2048) -> Optional[str]:
        """
        Generate text using Claude.

        Args:
            prompt: User prompt
            system: System instructions (optional)
            temperature: Temperature for generation
            model: Model to use (required)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text or None on error
        """
        if not model:
            raise ValueError("Model name is required for Claude")

        try:
            url = f"{self.base_url}/messages"
            headers = self._get_headers()

            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            # Add system prompt if provided
            if system:
                payload["system"] = system

            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            data = response.json()

            # Extract text from response
            if 'content' in data and len(data['content']) > 0:
                return data['content'][0].get('text', '')

            logger.error("Unexpected Claude response format")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Claude generation failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return None

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3,
             model: Optional[str] = None, max_tokens: int = 2048) -> Optional[str]:
        """
        Chat completion using Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Temperature for generation
            model: Model to use (required)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response or None on error
        """
        if not model:
            raise ValueError("Model name is required for Claude")

        try:
            url = f"{self.base_url}/messages"
            headers = self._get_headers()

            # Extract system prompt if present
            system_prompt = None
            claude_messages = []

            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

                if role == 'system':
                    system_prompt = content
                else:
                    claude_messages.append({
                        "role": role,
                        "content": content
                    })

            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": claude_messages
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            data = response.json()

            # Extract text from response
            if 'content' in data and len(data['content']) > 0:
                return data['content'][0].get('text', '')

            logger.error("Unexpected Claude response format")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Claude chat failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return None

    def generate_with_image(self, prompt: str, image_data: str,
                           system: Optional[str] = None, temperature: float = 0.1,
                           model: Optional[str] = None, max_tokens: int = 2048,
                           media_type: str = "image/jpeg") -> Optional[str]:
        """
        Generate text from image and prompt using Claude.

        Args:
            prompt: Text prompt
            image_data: Base64-encoded image data
            system: System instructions (optional)
            temperature: Temperature for generation
            model: Model to use (required)
            max_tokens: Maximum tokens to generate
            media_type: Media type of the image

        Returns:
            Generated text or None on error
        """
        if not model:
            raise ValueError("Model name is required for Claude")

        try:
            url = f"{self.base_url}/messages"
            headers = self._get_headers()

            # Build request with text and image in content blocks
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            # Add system prompt if provided
            if system:
                payload["system"] = system

            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT * 2)
            response.raise_for_status()

            data = response.json()

            # Extract text from response
            if 'content' in data and len(data['content']) > 0:
                return data['content'][0].get('text', '')

            logger.error("Unexpected Claude vision response format")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Claude vision generation failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return None
