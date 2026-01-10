"""Web page fetching and parsing utilities."""

import requests
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
from modulle.utils.http_client import fetch_url, create_session
from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)


class WebFetcher:
    """Fetches and parses web pages."""

    def __init__(self, session: Optional[requests.Session] = None):
        """
        Initialize web fetcher.

        Args:
            session: Optional requests session to use. If not provided,
                    a new session with retry logic will be created.
        """
        self.session = session or create_session()

    def fetch_page(self, url: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Fetch a web page and return parsed content.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            Dict with keys: url, title, text, html, status_code, headers
            Returns None on error
        """
        try:
            response = fetch_url(url, session=self.session, timeout=timeout)

            # Parse with BeautifulSoup using lxml parser (fallback to html.parser)
            try:
                soup = BeautifulSoup(response.content, 'lxml')
            except Exception:
                soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script, style, and navigation elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()

            # Extract title
            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Extract text content
            text = soup.get_text(separator='\n', strip=True)

            return {
                'url': response.url,
                'title': title,
                'text': text,
                'html': response.text,
                'status_code': response.status_code,
                'headers': dict(response.headers)
            }

        except Exception as e:
            logger.error(f"Failed to fetch page {url}: {e}")
            return None

    def fetch_article_content(self, url: str) -> Optional[str]:
        """
        Fetch main article content from a page.
        Uses heuristics to extract main content area.

        Args:
            url: Article URL to fetch

        Returns:
            Extracted article text or None on error
        """
        page_data = self.fetch_page(url)
        if not page_data:
            return None

        try:
            # Parse with BeautifulSoup
            try:
                soup = BeautifulSoup(page_data['html'], 'lxml')
            except Exception:
                soup = BeautifulSoup(page_data['html'], 'html.parser')

            # Try common article containers in order of preference
            article = soup.find('article')
            if article:
                return article.get_text(separator='\n', strip=True)

            # Try main content area
            main = soup.find('main')
            if main:
                return main.get_text(separator='\n', strip=True)

            # Try content div with common class names
            for class_name in ['content', 'article-content', 'post-content', 'entry-content']:
                content_div = soup.find('div', class_=class_name)
                if content_div:
                    return content_div.get_text(separator='\n', strip=True)

            # Fallback to full body text
            return page_data['text']

        except Exception as e:
            logger.error(f"Failed to extract article content from {url}: {e}")
            return None
