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

    def chat_with_tools(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Chat with function calling support.

        Gemini's function calling allows the model to call external functions
        when needed. The model returns function_call parts that your code should execute.

        Args:
            model: Model to use (must support function calling, e.g., gemini-1.5-flash)
            messages: List of message dicts with 'role' and 'content'
            tools: List of tool definitions in Gemini format
            temperature: Generation temperature (0.0 to 1.0)

        Returns:
            Dict with keys:
                - 'content': Generated text (if any)
                - 'tool_calls': List of tool calls (if model wants to call tools)
                - 'finish_reason': Why generation stopped ('stop' or 'tool_calls')
                - 'message': Full response data from Gemini

        Example:
            >>> tools = [{"name": "search_web", "description": "...", "parameters": {...}}]
            >>> result = client.chat_with_tools(model, messages, tools)
            >>> if result['tool_calls']:
            ...     for call in result['tool_calls']:
            ...         # Execute tool and add result to messages
            ...         pass
        """
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
                elif role == 'tool':
                    # Gemini format for function responses
                    contents.append({
                        "role": "function",
                        "parts": [{
                            "functionResponse": {
                                "name": msg.get('name', ''),
                                "response": {
                                    "result": content
                                }
                            }
                        }]
                    })
                elif role == 'assistant' and msg.get('tool_calls'):
                    # Convert our format to Gemini's function call format
                    function_call_parts = []
                    for tc in msg['tool_calls']:
                        function_call_parts.append({
                            "functionCall": {
                                "name": tc['name'],
                                "args": tc['arguments']
                            }
                        })
                    contents.append({
                        "role": "model",
                        "parts": function_call_parts
                    })
                else:
                    # Map assistant to model
                    gemini_role = 'model' if role == 'assistant' else 'user'
                    contents.append({
                        "role": gemini_role,
                        "parts": [{"text": content}]
                    })

            # Build tool declarations for Gemini
            function_declarations = []
            for tool in tools:
                function_declarations.append({
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": tool['parameters']
                })

            payload = {
                "contents": contents,
                "tools": [{
                    "functionDeclarations": function_declarations
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 4096
                }
            }

            if system_instruction:
                payload["systemInstruction"] = system_instruction

            logger.debug(f"Sending chat with tools request to Gemini model: {model}")
            logger.debug(f"Available tools: {[t['name'] for t in tools]}")

            response = requests.post(url, params=params, json=payload, timeout=REQUEST_TIMEOUT * 3)
            response.raise_for_status()

            data = response.json()

            # Parse response
            content_text = ""
            tool_calls = []

            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    for part in candidate['content']['parts']:
                        if 'text' in part:
                            content_text += part['text']
                        elif 'functionCall' in part:
                            func_call = part['functionCall']
                            logger.debug(f"Model requested function call: {func_call.get('name')}")
                            tool_calls.append({
                                'id': f"call_{len(tool_calls)}",  # Gemini doesn't provide IDs
                                'name': func_call.get('name', ''),
                                'arguments': func_call.get('args', {})
                            })

            finish_reason = 'tool_calls' if tool_calls else 'stop'

            logger.debug(f"Generated {len(content_text)} characters, finish_reason: {finish_reason}")

            return {
                'content': content_text if content_text else None,
                'tool_calls': tool_calls,
                'finish_reason': finish_reason,
                'message': data
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini chat with tools failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return {
                'content': None,
                'tool_calls': [],
                'finish_reason': 'error',
                'message': None
            }
        except Exception as e:
            logger.error(f"Unexpected error in Gemini chat with tools: {e}")
            return {
                'content': None,
                'tool_calls': [],
                'finish_reason': 'error',
                'message': None
            }

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
