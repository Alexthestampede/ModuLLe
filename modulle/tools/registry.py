"""Tool registry for managing and executing tools."""

from typing import Dict, List, Optional, Any
from .base import BaseTool
from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """
    Registry for managing and executing tools.

    The registry maintains a collection of available tools and provides
    methods to convert them to provider-specific formats and execute them.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(SearchWebTool(web))
        >>> registry.register(FetchPageTool(web))
        >>> tools = registry.to_ollama_format()
        >>> result = registry.execute("search_web", query="Python")
    """

    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """
        Register a tool in the registry.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If a tool with the same name is already registered
        """
        name = tool.get_name()
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, replacing")
        self._tools[name] = tool
        logger.debug(f"Registered tool: {name}")

    def unregister(self, tool_name: str):
        """
        Remove a tool from the registry.

        Args:
            tool_name: Name of tool to remove

        Raises:
            KeyError: If tool not found
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool not found: {tool_name}")
        del self._tools[tool_name]
        logger.debug(f"Unregistered tool: {tool_name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def execute(self, tool_name: str, **kwargs) -> str:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Parameters to pass to tool

        Returns:
            Tool execution result as string

        Raises:
            ValueError: If tool not found
            Exception: If tool execution fails
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        try:
            logger.info(f"Executing tool: {tool_name} with args: {kwargs}")
            result = tool.execute(**kwargs)
            logger.debug(f"Tool {tool_name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def to_ollama_format(self) -> List[Dict[str, Any]]:
        """
        Convert all tools to Ollama format.

        Returns:
            List of tool definitions in Ollama format
        """
        return [tool.to_ollama_schema() for tool in self._tools.values()]

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """
        Convert all tools to OpenAI format.

        Returns:
            List of tool definitions in OpenAI format
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def to_claude_format(self) -> List[Dict[str, Any]]:
        """
        Convert all tools to Claude format.

        Returns:
            List of tool definitions in Claude format
        """
        return [tool.to_claude_schema() for tool in self._tools.values()]

    def to_gemini_format(self) -> List[Dict[str, Any]]:
        """
        Convert all tools to Gemini format.

        Returns:
            List of tool definitions in Gemini format
        """
        return [tool.to_gemini_schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self._tools
