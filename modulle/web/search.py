"""Web search functionality with support for multiple backends."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum
from duckduckgo_search import DDGS
from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)


class SearchBackend(Enum):
    """Available search backends."""
    DUCKDUCKGO = "duckduckgo"
    SERPAPI = "serpapi"


class BaseSearcher(ABC):
    """Abstract base class for search implementations."""

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search the web.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of result dicts with keys: title, url, snippet
        """
        pass


class DuckDuckGoSearcher(BaseSearcher):
    """DuckDuckGo search implementation."""

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search using DuckDuckGo.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of search results with title, url, and snippet
        """
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)

            return [
                {
                    'title': r.get('title', ''),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', '')
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"DuckDuckGo search failed for query '{query}': {e}")
            return []


class SerpAPISearcher(BaseSearcher):
    """SerpAPI search implementation (requires API key)."""

    def __init__(self, api_key: str):
        """
        Initialize SerpAPI searcher.

        Args:
            api_key: SerpAPI API key
        """
        self.api_key = api_key

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search using SerpAPI.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of search results with title, url, and snippet
        """
        try:
            # Import serpapi only if SerpAPI is actually used
            from serpapi import GoogleSearch

            params = {
                "q": query,
                "num": max_results,
                "api_key": self.api_key
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            organic_results = results.get('organic_results', [])
            return [
                {
                    'title': r.get('title', ''),
                    'url': r.get('link', ''),
                    'snippet': r.get('snippet', '')
                }
                for r in organic_results[:max_results]
            ]

        except ImportError:
            logger.error("SerpAPI library not installed. Install with: pip install google-search-results")
            return []
        except Exception as e:
            logger.error(f"SerpAPI search failed for query '{query}': {e}")
            return []


def create_searcher(
    backend: SearchBackend = SearchBackend.DUCKDUCKGO,
    api_key: Optional[str] = None
) -> BaseSearcher:
    """
    Factory function to create a search backend instance.

    Args:
        backend: Search backend to use
        api_key: API key for paid search services (e.g., SerpAPI)

    Returns:
        Configured searcher instance

    Raises:
        ValueError: If backend is unknown or required API key is missing
    """
    if backend == SearchBackend.DUCKDUCKGO:
        return DuckDuckGoSearcher()
    elif backend == SearchBackend.SERPAPI:
        if not api_key:
            raise ValueError("SerpAPI backend requires api_key parameter")
        return SerpAPISearcher(api_key)
    else:
        raise ValueError(f"Unknown search backend: {backend}")
