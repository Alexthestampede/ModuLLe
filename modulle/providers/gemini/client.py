"""
Google Gemini API client for RSS Feed Processor

This module provides integration with Google's Gemini API using the
generativelanguage.googleapis.com endpoint.
"""
import requests
from typing import Optional, List, Dict, Any
from modulle.utils.logging_config import get_logger
from modulle.config import REQUEST_TIMEOUT
from modulle.base import BaseAIClient

logger = get_logger(__name__.replace("modulle.providers.", ""))


class GeminiClient(BaseAIClient):
    """
    Client for interacting with Google Gemini API.

    Gemini uses a different API format than OpenAI, with 'contents' and 'parts'
    structure instead of messages.
    """

    def __init__(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        """
        Initialize Gemini client.

        Args:
            api_key: Google API key for Gemini
            base_url: Gemini API base URL
        """
        if not api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

        self.api_key = api_key
        self.base_url = base_url.rstrip('/')

    def health_check(self) -> bool:
        """
        Check if Gemini API is available.

        Returns:
            True if API is available, False otherwise
        """
        try:
            models = self.list_models()
            if models:
                logger.info("Gemini API is available")
                return True
            return False
        except Exception as e:
            logger.error(f"Gemini API health check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        """
        List available Gemini models.

        Returns:
            List of model names
        """
        try:
            url = f"{self.base_url}/models"
            params = {"key": self.api_key}

            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            data = response.json()
            models = [model.get('name', '').replace('models/', '') for model in data.get('models', [])]

            logger.info(f"Found {len(models)} Gemini models")
            return models

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list Gemini models: {e}")
            return []

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.3, model: Optional[str] = None) -> Optional[str]:
        """
        Generate text using Gemini.

        Args:
            prompt: User prompt
            system: System instructions (optional)
            temperature: Temperature for generation
            model: Model to use (required)

        Returns:
            Generated text or None on error
        """
        if not model:
            raise ValueError("Model name is required for Gemini")

        try:
            url = f"{self.base_url}/models/{model}:generateContent"
            params = {"key": self.api_key}

            # Build request payload
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 2048
                }
            }

            # Add system instruction if provided
            if system:
                payload["systemInstruction"] = {
                    "parts": [{"text": system}]
                }

            response = requests.post(url, params=params, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            data = response.json()

            # Extract text from response
            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        return parts[0]['text']

            logger.error("Unexpected Gemini response format")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini generation failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return None

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3,
             model: Optional[str] = None) -> Optional[str]:
        """
        Chat completion using Gemini.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Temperature for generation
            model: Model to use (required)

        Returns:
            Generated response or None on error
        """
        if not model:
            raise ValueError("Model name is required for Gemini")

        try:
            url = f"{self.base_url}/models/{model}:generateContent"
            params = {"key": self.api_key}

            # Convert messages to Gemini format
            contents = []
            system_instruction = None

            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

                if role == 'system':
                    system_instruction = {"parts": [{"text": content}]}
                else:
                    # Map assistant to model
                    gemini_role = 'model' if role == 'assistant' else 'user'
                    contents.append({
                        "role": gemini_role,
                        "parts": [{"text": content}]
                    })

            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 2048
                }
            }

            if system_instruction:
                payload["systemInstruction"] = system_instruction

            response = requests.post(url, params=params, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            data = response.json()

            # Extract text from response
            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        return parts[0]['text']

            logger.error("Unexpected Gemini response format")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini chat failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return None

    def generate_with_image(self, prompt: str, image_data: str,
                           system: Optional[str] = None, temperature: float = 0.1,
                           model: Optional[str] = None, mime_type: str = "image/jpeg") -> Optional[str]:
        """
        Generate text from image and prompt using Gemini.

        Args:
            prompt: Text prompt
            image_data: Base64-encoded image data
            system: System instructions (optional)
            temperature: Temperature for generation
            model: Model to use (required)
            mime_type: MIME type of the image

        Returns:
            Generated text or None on error
        """
        if not model:
            raise ValueError("Model name is required for Gemini")

        try:
            url = f"{self.base_url}/models/{model}:generateContent"
            params = {"key": self.api_key}

            # Build request with text and image
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_data
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 2048
                }
            }

            # Add system instruction if provided
            if system:
                payload["systemInstruction"] = {
                    "parts": [{"text": system}]
                }

            response = requests.post(url, params=params, json=payload, timeout=REQUEST_TIMEOUT * 2)
            response.raise_for_status()

            data = response.json()

            # Extract text from response
            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        return parts[0]['text']

            logger.error("Unexpected Gemini vision response format")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini vision generation failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return None
