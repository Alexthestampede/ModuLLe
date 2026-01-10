"""Tool calling framework for ModuLLe.

This module provides a framework for creating tools that LLMs can call autonomously.
Tools allow LLMs to access external functionality like web search, file operations,
calculations, etc.

Example:
    >>> from modulle.tools import BaseTool, ToolRegistry
    >>> from modulle.web.tools import SearchWebTool, FetchPageTool
    >>> from modulle.web import WebAccessor
    >>>
    >>> # Create tools
    >>> web = WebAccessor()
    >>> registry = ToolRegistry()
    >>> registry.register(SearchWebTool(web))
    >>> registry.register(FetchPageTool(web))
    >>>
    >>> # Use with LLM
    >>> client.chat_with_tools(messages, tools=registry.to_ollama_format())
"""

from .base import BaseTool
from .registry import ToolRegistry

__all__ = ['BaseTool', 'ToolRegistry']
