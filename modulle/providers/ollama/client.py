"""
Base Ollama client for RSS Feed Processor
"""

from typing import Any, Dict, List, Optional

import requests

from modulle.config import OLLAMA_BASE_URL, REQUEST_TIMEOUT
from modulle.utils.logging_config import get_logger
from modulle.utils.response_cleaner import clean_response

logger = get_logger(__name__.replace("modulle.providers.", ""))


class OllamaClient:
    """
    Base client for interacting with Ollama API.
    """

    def __init__(self, base_url=OLLAMA_BASE_URL, request_timeout: Optional[int] = None):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server base URL
            request_timeout: Timeout in seconds for API requests.
                Generation calls use 3x this value. Defaults to the
                REQUEST_TIMEOUT config value (MODULLE_REQUEST_TIMEOUT env var)
                when not provided.
        """
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api"
        self.request_timeout = request_timeout or REQUEST_TIMEOUT

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
            response = requests.get(f"{self.api_url}/tags", timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
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
                "options": {"temperature": temperature},
            }

            if system:
                payload["system"] = system

            if images:
                payload["images"] = images

            logger.debug(f"Sending request to Ollama model: {model}")
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=self.request_timeout * 3,  # Generation needs longer than health/list calls
            )

            # Check for errors and try to get detailed error message from Ollama
            if not response.ok:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", f"HTTP {response.status_code}")
                    logger.error(f"Ollama generation failed: {error_msg}")
                except Exception:
                    logger.error(f"Ollama generation failed: HTTP {response.status_code}")
                return None

            data = response.json()
            generated_text = clean_response(data.get("response", ""))

            logger.debug(f"Generated {len(generated_text or '')} characters")
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
                "options": {"temperature": temperature},
            }

            logger.debug(f"Sending chat request to Ollama model: {model}")
            response = requests.post(
                f"{self.api_url}/chat", json=payload, timeout=self.request_timeout * 3
            )

            # Check for errors and try to get detailed error message from Ollama
            if not response.ok:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", f"HTTP {response.status_code}")
                    logger.error(f"Ollama chat failed: {error_msg}")
                except Exception:
                    logger.error(f"Ollama chat failed: HTTP {response.status_code}")
                return None

            data = response.json()
            message = data.get("message", {})
            content = clean_response(message.get("content", ""))

            logger.debug(f"Generated {len(content or '')} characters")
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama chat failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Ollama chat: {e}")
            return None

    def chat_with_tools(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Chat with tool calling support.

        Ollama supports tool calling in compatible models. The model can
        choose to call tools, and the results can be fed back into the
        conversation.

        Args:
            model: Model name to use (must support tool calling)
            messages: List of message dicts with 'role' and 'content'
            tools: List of tool definitions in Ollama format
            temperature: Generation temperature

        Returns:
            Dict with keys:
                - 'content': Generated text (if any)
                - 'tool_calls': List of tool calls (if model wants to call tools)
                - 'finish_reason': Why generation stopped ('stop' or 'tool_calls')
                - 'message': Full message dict from Ollama

        Example:
            >>> tools = [{"type": "function", "function": {...}}]
            >>> result = client.chat_with_tools(model, messages, tools)
            >>> if result['tool_calls']:
            ...     for call in result['tool_calls']:
            ...         # Execute tool and add result to messages
            ...         pass
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"temperature": temperature},
            }

            logger.debug(f"Sending chat with tools request to Ollama model: {model}")
            logger.debug(f"Available tools: {[t['function']['name'] for t in tools]}")

            response = requests.post(
                f"{self.api_url}/chat", json=payload, timeout=self.request_timeout * 3
            )

            # Check for errors
            if not response.ok:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", f"HTTP {response.status_code}")
                    logger.error(f"Ollama chat with tools failed: {error_msg}")
                except Exception:
                    logger.error(f"Ollama chat with tools failed: HTTP {response.status_code}")
                return {
                    "content": None,
                    "tool_calls": [],
                    "finish_reason": "error",
                    "message": None,
                }

            data = response.json()
            message = data.get("message", {})
            content = clean_response(message.get("content", ""))
            tool_calls_raw = message.get("tool_calls", [])

            # Parse tool calls
            tool_calls = []
            if tool_calls_raw:
                logger.debug(f"Model requested {len(tool_calls_raw)} tool call(s)")
                for tc in tool_calls_raw:
                    function = tc.get("function", {})
                    tool_calls.append(
                        {
                            "id": tc.get("id", f"call_{len(tool_calls)}"),
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", {}),
                        }
                    )

            # Determine finish reason
            finish_reason = "tool_calls" if tool_calls else "stop"

            logger.debug(f"Generated {len(content)} characters, finish_reason: {finish_reason}")

            return {
                "content": content if content else None,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
                "message": message,
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama chat with tools failed: {e}")
            return {"content": None, "tool_calls": [], "finish_reason": "error", "message": None}
        except Exception as e:
            logger.error(f"Unexpected error in Ollama chat with tools: {e}")
            return {"content": None, "tool_calls": [], "finish_reason": "error", "message": None}

    def system_one(
        self,
        model: str,
        state: Any,
        questions: Dict[str, Dict[str, Any]],
        images: Optional[List[str]] = None,
        keep_alive: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Decision-model API (e.g. clef / clef-flash) via Ollama's /v1/systemone endpoint.

        Decision models score every option of every question jointly in a single
        non-autoregressive pass. Requires Ollama >= 0.35.1.

        Args:
            model: Model name to use (e.g. 'clef-flash')
            state: Text (or JSON-serializable object/array) to judge
            questions: Dict of 1-64 named questions. Each question needs a 'type'
                ('noul', 'choice', or 'score'), 'instructions', and usually
                'criteria' (choice: option->description, 2-26 options;
                score: list of level descriptions lowest-first, 2-26 levels;
                noul: optionally {"true": "...", "false": "..."})
            images: Optional list of base64-encoded PNG/JPEG/WebP images shared
                by all questions (URLs and data URLs are not supported)
            keep_alive: Optional keep-alive duration for the model
            timeout: Request timeout in seconds (defaults to request_timeout * 3)

        Returns:
            Dict with keys:
                - 'model': Model name
                - 'answers': Dict mapping each question name to its typed answer
                    (noul: {'type', 'noul'}, choice: {'type', 'choice',
                    'probabilities', 'confidence'}, score: {'type', 'score',
                    'legend', 'probabilities', 'confidence'})
                - 'usage': Token usage dict

            None on error.

        Example:
            >>> result = client.system_one(
            ...     model='clef-flash',
            ...     state='Checkout failing for all customers',
            ...     questions={
            ...         'urgent': {'type': 'noul', 'instructions': 'Is this urgent?'},
            ...         'team': {'type': 'choice', 'instructions': 'Which team?',
            ...                  'criteria': {'billing': 'Payments', 'technical': 'Outages'}}
            ...     }
            ... )
            >>> result['answers']['team']['choice']
            'technical'
        """
        try:
            payload: Dict[str, Any] = {"model": model, "state": state, "questions": questions}

            if images:
                payload["images"] = images
            if keep_alive:
                payload["keep_alive"] = keep_alive

            logger.debug(f"Sending systemone request to decision model: {model}")
            response = requests.post(
                f"{self.base_url}/v1/systemone",
                json=payload,
                timeout=timeout if timeout is not None else self.request_timeout * 3,
            )

            # Check for errors and try to get detailed error message from Ollama
            if not response.ok:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", f"HTTP {response.status_code}")
                    logger.error(f"Ollama systemone failed: {error_msg}")
                except Exception:
                    logger.error(f"Ollama systemone failed: HTTP {response.status_code}")
                return None

            data = response.json()
            answers = data.get("answers", {})
            logger.debug(f"Got {len(answers)} answer(s) from decision model")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama systemone failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Ollama systemone: {e}")
            return None
