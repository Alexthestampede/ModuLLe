"""Web access tools for LLM tool calling."""

from typing import Dict, Any
from modulle.tools.base import BaseTool
from .accessor import WebAccessor
from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)


class SearchWebTool(BaseTool):
    """
    Tool that allows LLMs to search the web.

    This tool enables LLMs to autonomously search for information
    when they need current data or external knowledge.
    """

    def __init__(self, web_accessor: WebAccessor):
        """
        Initialize search tool.

        Args:
            web_accessor: WebAccessor instance to use for searching
        """
        self.web = web_accessor

    def get_name(self) -> str:
        """Return tool name."""
        return "search_web"

    def get_description(self) -> str:
        """Return tool description."""
        return (
            "Search the web for information. Use this tool when you need to find "
            "current information, news, articles, documentation, or any external knowledge. "
            "Returns a list of search results with titles, URLs, and brief snippets. "
            "Each result provides enough context to decide if you should fetch the full page."
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Return parameter schema."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and use relevant keywords."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 10)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        }

    def execute(self, query: str, max_results: int = 5) -> str:
        """
        Execute web search.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            Formatted search results as string
        """
        try:
            # Ensure max_results is within bounds
            max_results = min(max(1, max_results), 10)

            logger.info(f"Searching web for: {query}")
            results = self.web.search_web(query, max_results)

            if not results:
                return f"No results found for query: {query}"

            # Format results for LLM
            formatted = f"Search results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                formatted += f"{i}. {result['title']}\n"
                formatted += f"   URL: {result['url']}\n"
                formatted += f"   Snippet: {result['snippet']}\n\n"

            logger.info(f"Found {len(results)} results")
            return formatted

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Error searching for '{query}': {str(e)}"


class FetchPageTool(BaseTool):
    """
    Tool that allows LLMs to fetch and read web pages.

    This tool enables LLMs to retrieve the full content of web pages
    they've discovered through search or other means.
    """

    def __init__(self, web_accessor: WebAccessor):
        """
        Initialize fetch tool.

        Args:
            web_accessor: WebAccessor instance to use for fetching
        """
        self.web = web_accessor

    def get_name(self) -> str:
        """Return tool name."""
        return "fetch_page"

    def get_description(self) -> str:
        """Return tool description."""
        return (
            "Fetch and return the full content of a web page. Use this tool when you need "
            "to read the complete text of a specific URL, such as an article, documentation, "
            "or blog post. The content is returned as clean text suitable for analysis. "
            "Note: Content may be truncated if very long to fit within context limits."
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Return parameter schema."""
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the web page to fetch. Must be a valid HTTP/HTTPS URL."
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "markdown"],
                    "description": "Output format: 'text' for plain text, 'markdown' for formatted markdown (default: text)",
                    "default": "text"
                }
            },
            "required": ["url"]
        }

    def execute(self, url: str, format: str = "text") -> str:
        """
        Execute page fetch.

        Args:
            url: URL to fetch
            format: Output format (text or markdown)

        Returns:
            Page content as string
        """
        try:
            logger.info(f"Fetching page: {url}")

            # Validate format
            if format not in ["text", "markdown"]:
                format = "text"

            content = self.web.fetch_page(url, format=format)

            if not content:
                return f"Failed to fetch content from {url}. The page may be unavailable or blocked."

            # Truncate if too long (leave room for other context)
            max_length = 8000
            if len(content) > max_length:
                content = content[:max_length]
                truncated_msg = f"\n\n[Content truncated to {max_length} characters]"
                content += truncated_msg

            logger.info(f"Successfully fetched {len(content)} characters from {url}")
            return content

        except Exception as e:
            logger.error(f"Fetch failed for {url}: {e}")
            return f"Error fetching {url}: {str(e)}"
