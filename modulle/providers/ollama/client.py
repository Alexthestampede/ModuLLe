"""
Base Ollama client for RSS Feed Processor
"""
import requests
from modulle.utils.logging_config import get_logger
from modulle.config import OLLAMA_BASE_URL, REQUEST_TIMEOUT

logger = get_logger(__name__.replace("modulle.providers.", ""))


class OllamaClient:
    """
    Base client for interacting with Ollama API.
    """

    def __init__(self, base_url=OLLAMA_BASE_URL):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server base URL
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"

    def health_check(self):
        """
        Check if Ollama server is available.

        Returns:
            True if server is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Ollama server is available")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama server is not available: {e}")
            return False

    def list_models(self):
        """
        List available models on the Ollama server.

        Returns:
            List of model names, or empty list on error
        """
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            logger.info(f"Available models: {models}")
            return models
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def generate(self, model, prompt, system=None, temperature=0.3, images=None):
        """
        Generate text using Ollama.

        Args:
            model: Model name to use
            prompt: User prompt
            system: System prompt (optional)
            temperature: Generation temperature
            images: List of base64-encoded images (for vision models)

        Returns:
            Generated text, or None on error
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }

            if system:
                payload["system"] = system

            if images:
                payload["images"] = images

            logger.debug(f"Sending request to Ollama model: {model}")
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=REQUEST_TIMEOUT * 3  # Longer timeout for generation
            )

            # Check for errors and try to get detailed error message from Ollama
            if not response.ok:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', f"HTTP {response.status_code}")
                    logger.error(f"Ollama generation failed: {error_msg}")
                except Exception:
                    logger.error(f"Ollama generation failed: HTTP {response.status_code}")
                return None

            data = response.json()
            generated_text = data.get('response', '').strip()

            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama generation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Ollama generation: {e}")
            return None

    def chat(self, model, messages, temperature=0.3):
        """
        Chat using Ollama chat API.

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
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }

            logger.debug(f"Sending chat request to Ollama model: {model}")
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=REQUEST_TIMEOUT * 3
            )

            # Check for errors and try to get detailed error message from Ollama
            if not response.ok:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', f"HTTP {response.status_code}")
                    logger.error(f"Ollama chat failed: {error_msg}")
                except Exception:
                    logger.error(f"Ollama chat failed: HTTP {response.status_code}")
                return None

            data = response.json()
            message = data.get('message', {})
            content = message.get('content', '').strip()

            logger.debug(f"Generated {len(content)} characters")
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama chat failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Ollama chat: {e}")
            return None
