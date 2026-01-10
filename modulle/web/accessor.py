"""High-level interface for web access."""

from typing import Optional, List, Dict, Any
from .fetcher import WebFetcher
from .search import create_searcher, SearchBackend
from .converter import HTMLConverter
from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)


class WebAccessor:
    """
    High-level interface for web access (direct method approach).

    This class provides direct methods for fetching web content and searching.
    Users explicitly call these methods and use LLMs to process the results.

    Example:
        >>> web = WebAccessor()
        >>> results = web.search_web("Python programming", max_results=5)
        >>> content = web.fetch_page(results[0]['url'])
        >>> # Use LLM to process content...
    """

    def __init__(
        self,
        search_backend: SearchBackend = SearchBackend.DUCKDUCKGO,
        search_api_key: Optional[str] = None
    ):
        """
        Initialize web accessor.

        Args:
            search_backend: Search backend to use (default: DuckDuckGo)
            search_api_key: API key for paid search services (e.g., SerpAPI)
        """
        self.fetcher = WebFetcher()
        self.searcher = create_searcher(search_backend, search_api_key)
        self.converter = HTMLConverter()

    def fetch_page(self, url: str, format: str = 'text') -> Optional[str]:
        """
        Fetch web page content.

        Args:
            url: URL to fetch
            format: Output format - 'text', 'markdown', 'html', or 'full'
                   'text': Clean text without HTML tags
                   'markdown': Markdown formatted content
                   'html': Raw HTML content
                   'full': Complete page data dict

        Returns:
            Page content in requested format, or None on error
        """
        page_data = self.fetcher.fetch_page(url)
        if not page_data:
            return None

        if format == 'text':
            return page_data['text']
        elif format == 'markdown':
            return self.converter.html_to_markdown(page_data['html'])
        elif format == 'html':
            return page_data['html']
        elif format == 'full':
            return page_data
        else:
            logger.warning(f"Unknown format '{format}', defaulting to text")
            return page_data['text']

    def fetch_article(self, url: str, format: str = 'text') -> Optional[str]:
        """
        Fetch main article content from a page using heuristics.

        Args:
            url: Article URL to fetch
            format: Output format - 'text' or 'markdown'

        Returns:
            Article content in requested format, or None on error
        """
        article_text = self.fetcher.fetch_article_content(url)
        if not article_text:
            return None

        if format == 'markdown':
            # Fetch full page data to get HTML
            page_data = self.fetcher.fetch_page(url)
            if page_data:
                return self.converter.html_to_markdown(page_data['html'])

        return article_text

    def search_web(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search the web.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 10)

        Returns:
            List of search results, each containing:
                - title: Page title
                - url: Page URL
                - snippet: Brief text excerpt
        """
        return self.searcher.search(query, max_results)

    def search_and_fetch(
        self,
        query: str,
        num_pages: int = 3,
        format: str = 'text',
        max_content_length: int = 5000
    ) -> List[Dict[str, str]]:
        """
        Search and fetch content from top results.

        This is a convenience method that combines search and fetch operations.

        Args:
            query: Search query string
            num_pages: Number of top results to fetch (default: 3)
            format: Content format - 'text' or 'markdown' (default: 'text')
            max_content_length: Maximum characters per page (default: 5000)

        Returns:
            List of dicts with keys: url, title, content
        """
        search_results = self.search_web(query, max_results=num_pages)

        fetched_content = []
        for result in search_results[:num_pages]:
            content = self.fetch_page(result['url'], format=format)
            if content:
                # Truncate content to avoid overwhelming LLM context
                if len(content) > max_content_length:
                    content = content[:max_content_length]

                fetched_content.append({
                    'url': result['url'],
                    'title': result['title'],
                    'content': content
                })

        return fetched_content
