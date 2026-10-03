"""SearXNG search backend for ModuLLe.

SearXNG is a self-hosted metasearch engine. This module talks to a SearXNG
instance's JSON API (requires the instance to allow ``format=json`` requests
for the ``search`` endpoint).

Example:
    >>> from modulle.web.searxng import SearxngSearcher, SearchSearxngTool
    >>> searcher = SearxngSearcher('http://localhost:8888')
    >>> results = searcher.search('Python async', max_results=5)
    >>> # Or expose it to the LLM as a tool:
    >>> tool = SearchSearxngTool(searcher)
    >>> registry.register(tool)
"""

import os
from typing import Any, Dict, List, Optional

import requests

from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)

# Dummy default — users must point this at their own instance in settings.
DEFAULT_SEARXNG_URL = "http://localhost:8888"


class SearxngSearcher:
    """Search backend using a SearXNG instance's JSON API.

    Attributes:
        base_url: Base URL of the SearXNG instance.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 10,
        categories: str = "general",
        language: str = "",
        safesearch: int = 1,
    ):
        """Initialize SearXNG searcher.

        Args:
            base_url: Base URL of the SearXNG instance. Falls back to the
                SEARXNG_BASE_URL environment variable, then
                http://localhost:8888.
            timeout: Request timeout in seconds.
            categories: SearXNG category filter (e.g. 'general', 'images',
                'news', 'it', 'science'). Empty string for no filter.
            language: ISO language code filter (e.g. 'en', 'it'). Empty
                string for automatic.
            safesearch: 0 = off, 1 = moderate, 2 = strict.
        """
        self.base_url = (
            base_url or os.getenv("SEARXNG_BASE_URL", "") or DEFAULT_SEARXNG_URL
        ).rstrip("/")
        # Guard against the common misconfiguration of a full search endpoint
        if self.base_url.endswith("/search"):
            self.base_url = self.base_url[: -len("/search")]
        self.timeout = timeout
        self.categories = categories or ""
        self.language = language or ""
        self.safesearch = int(safesearch) if safesearch is not None else 1

    def is_available(self) -> bool:
        """Check that the SearXNG instance responds."""
        try:
            resp = requests.get(
                self.base_url,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
            return resp.status_code < 500
        except requests.RequestException:
            return False

    def search(
        self,
        query: str,
        max_results: int = 10,
        categories: Optional[str] = None,
        language: Optional[str] = None,
        safesearch: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search using the SearXNG JSON API.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            categories: Override the default category for this call.
            language: Override the default language for this call.
            safesearch: Override the default safesearch for this call.

        Returns:
            List of result dicts with keys: title, url, snippet (and
            'engine' / 'score' when provided by the server).

        Raises:
            requests.RequestException: On connection or HTTP errors.
            ValueError: If the instance returned non-JSON (format=json not
                enabled server-side).
        """
        params = {
            "q": query,
            "format": "json",
            "safesearch": safesearch if safesearch is not None else self.safesearch,
        }
        cats = categories if categories is not None else self.categories
        if cats:
            params["categories"] = cats
        lang = language if language is not None else self.language
        if lang:
            params["language"] = lang

        resp = requests.get(
            f"{self.base_url}/search",
            params=params,
            timeout=self.timeout,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError as e:
            raise ValueError(
                f"SearXNG at {self.base_url} did not return JSON. "
                "Enable 'formats: [html, json]' in the instance's settings.yml."
            ) from e

        results = []
        for r in data.get("results", [])[:max_results]:
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "engine": r.get("engine", ""),
                    "score": r.get("score", 0),
                }
            )
        return results
