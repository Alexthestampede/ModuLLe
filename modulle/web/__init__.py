"""Web access module for ModuLLe.

This module provides web content fetching and search capabilities
that work with any ModuLLe AI provider.

Example:
    >>> from modulle.web import WebAccessor
    >>> from modulle import create_ai_client
    >>>
    >>> # Create web accessor
    >>> web = WebAccessor()
    >>>
    >>> # Search the web
    >>> results = web.search_web("Python async programming", max_results=5)
    >>>
    >>> # Fetch page content
    >>> content = web.fetch_page(results[0]['url'], format='markdown')
    >>>
    >>> # Use AI to process content
    >>> _, processor, _ = create_ai_client('openai', text_model='gpt-4o-mini')
    >>> summary = processor.generate(
    ...     prompt=f"Summarize: {content[:3000]}",
    ...     temperature=0.3
    ... )
"""

from .accessor import WebAccessor
from .search import SearchBackend

__all__ = ['WebAccessor', 'SearchBackend']
