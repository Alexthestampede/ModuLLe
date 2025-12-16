"""
LM Studio client for RSS Feed Processor

LM Studio provides an OpenAI-compatible API, so this client uses the
OpenAI chat completions format for all operations.
"""
import requests
from typing import Optional, List, Dict
from modulle.utils.logging_config import get_logger
from modulle.config import LM_STUDIO_BASE_URL, REQUEST_TIMEOUT
from modulle.base import BaseAIClient

logger = get_logger(__name__.replace("modulle.providers.", ""))


class LMStudioClient(BaseAIClient):
    """
    Client for interacting with LM Studio's OpenAI-compatible API.

    LM Studio provides a local OpenAI-compatible server that supports
    the /v1/chat/completions endpoint.
    """

    def __init__(self, base_url=LM_STUDIO_BASE_URL):
        """
        Initialize LM Studio client.

        Args:
            base_url: LM Studio server base URL
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/v1"

    def health_check(self) -> bool:
        """
        Check if LM Studio server is available.

        Returns:
            True if server is available, False otherwise
        """
        try:
            response = requests.get(f"{self.api_url}/models", timeout=5)
            response.raise_for_status()
            logger.info("LM Studio server is available")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"LM Studio server is not available: {e}")
            return False

    def list_models(self) -> List[str]:
        """
        List available models on the LM Studio server.

        Returns:
            List of model names, or empty list on error
        """
        try:
            response = requests.get(f"{self.api_url}/models", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            # OpenAI format: {"data": [{"id": "model-name", ...}, ...]}
            models = [model['id'] for model in data.get('data', [])]
            logger.info(f"Available models: {models}")
            return models

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
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
        Generate text using LM Studio.

        This method converts the generate-style API to OpenAI's chat format.

        Args:
            model: Model name to use
            prompt: User prompt
            system: System prompt (optional)
            temperature: Generation temperature
            images: List of base64-encoded images (for vision models)
                   Note: LM Studio vision support may vary by model

        Returns:
            Generated text, or None on error
        """
        try:
            # Build messages in OpenAI format
            messages = []

            if system:
                messages.append({
                    "role": "system",
                    "content": system
                })

            # For vision models, include images in content array
            if images:
                # Vision format: content is an array of text + image parts
                content = [{"type": "text", "text": prompt}]
                for image_data in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    })
                messages.append({
                    "role": "user",
                    "content": content
                })
            else:
                # Text-only format
                messages.append({
                    "role": "user",
                    "content": prompt
                })

            # Call the chat API
            return self.chat(model, messages, temperature)

        except Exception as e:
            logger.error(f"Unexpected error in LM Studio generation: {e}")
            return None

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3
    ) -> Optional[str]:
        """
        Chat using LM Studio's OpenAI-compatible chat API.

        Args:
            model: Model name to use
            messages: List of message dicts with 'role' and 'content'
            temperature: Generation temperature

        Returns:
            Generated response, or None on error
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": False
            }

            logger.debug(f"Sending request to LM Studio model: {model}")
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                timeout=REQUEST_TIMEOUT * 3  # Longer timeout for generation
            )
            response.raise_for_status()

            data = response.json()

            # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
            choices = data.get('choices', [])
            if not choices:
                logger.error("No choices in LM Studio response")
                return None

            content = choices[0].get('message', {}).get('content', '').strip()

            logger.debug(f"Generated {len(content)} characters")
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"LM Studio chat failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in LM Studio chat: {e}")
            return None
