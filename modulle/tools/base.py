"""Base classes for tool calling."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Tools are functions that LLMs can call to access external functionality.
    Each tool must define its name, description, parameters, and execution logic.
    """

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the tool name.

        This is used by the LLM to identify which tool to call.
        Should be a valid function name (lowercase, underscores).

        Returns:
            Tool name (e.g., "search_web", "fetch_page")
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Return a description of what the tool does.

        This helps the LLM understand when to use the tool.
        Should be clear and concise.

        Returns:
            Human-readable description of tool functionality
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return JSON Schema for tool parameters.

        Defines what arguments the tool accepts.
        Must be valid JSON Schema format.

        Returns:
            Dict containing JSON Schema with type, properties, required fields

        Example:
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool parameters as defined in get_parameters()

        Returns:
            Tool execution result as a string (will be returned to LLM)
        """
        pass

    def to_ollama_schema(self) -> Dict[str, Any]:
        """
        Convert tool to Ollama function calling format.

        Returns:
            Dict in Ollama tool format
        """
        return {
            "type": "function",
            "function": {
                "name": self.get_name(),
                "description": self.get_description(),
                "parameters": self.get_parameters()
            }
        }

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Convert tool to OpenAI function calling format.

        Returns:
            Dict in OpenAI tool format
        """
        return {
            "type": "function",
            "function": {
                "name": self.get_name(),
                "description": self.get_description(),
                "parameters": self.get_parameters()
            }
        }

    def to_claude_schema(self) -> Dict[str, Any]:
        """
        Convert tool to Claude tool format.

        Returns:
            Dict in Claude tool format
        """
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "input_schema": self.get_parameters()
        }

    def to_gemini_schema(self) -> Dict[str, Any]:
        """
        Convert tool to Gemini function calling format.

        Returns:
            Dict in Gemini tool format
        """
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "parameters": self.get_parameters()
        }
