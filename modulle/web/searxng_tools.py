"""Tool wrappers for SearXNG search, for LLM tool calling."""

from typing import Dict, Any, Optional

from modulle.tools.base import BaseTool
from modulle.utils.logging_config import get_logger
from .searxng import SearxngSearcher

logger = get_logger(__name__)


class SearchSearxngTool(BaseTool):
    """Tool that lets LLMs search via a SearXNG instance."""

    def __init__(self, searcher: Optional[SearxngSearcher] = None):
        """Initialize the tool.

        Args:
            searcher: SearxngSearcher instance. One is created with defaults
                if omitted.
        """
        self.searcher = searcher or SearxngSearcher()

    def get_name(self) -> str:
        """Return tool name."""
        return "search_web_searxng"

    def get_description(self) -> str:
        """Return tool description."""
        return (
            "Search the web via a SearXNG metasearch instance. "
            "Use for current information, news, documentation, or external "
            "knowledge. Returns a list of results with titles, URLs and "
            "snippets."
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
                    "description": "Maximum number of results to return (default: 5, max: 15)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 15
                },
                "categories": {
                    "type": "string",
                    "description": "Optional SearXNG category filter, e.g. 'general', "
                                   "'images', 'news', 'it', 'science'",
                    "default": ""
                },
                "language": {
                    "type": "string",
                    "description": "Optional ISO language code filter, e.g. 'en', 'it'",
                    "default": ""
                }
            },
            "required": ["query"]
        }

    def execute(self, query: str, max_results: int = 5,
                categories: str = "", language: str = "") -> str:
        """Execute the search and format results for the LLM.

        Args:
            query: Search query string.
            max_results: Maximum number of results.
            categories: Optional category override.
            language: Optional language override.

        Returns:
            Formatted search results as string.
        """
        try:
            max_results = min(max(1, int(max_results)), 15)
            logger.info(f"SearXNG search for: {query}")
            results = self.searcher.search(
                query,
                max_results=max_results,
                categories=categories or None,
                language=language or None,
            )
            if not results:
                return f"No results found for query: {query}"

            formatted = f"SearXNG results for '{query}':\n\n"
            for i, r in enumerate(results, 1):
                formatted += f"{i}. {r['title']}\n"
                formatted += f"   URL: {r['url']}\n"
                formatted += f"   Snippet: {r['snippet']}\n\n"
            logger.info(f"Found {len(results)} results")
            return formatted
        except Exception as e:
            logger.error(f"SearXNG search failed: {e}")
            return f"Error searching for '{query}': {str(e)}"