"""HTML to text/markdown conversion utilities."""

import html2text
from typing import Optional


class HTMLConverter:
    """Converts HTML to clean text formats for LLM consumption."""

    def __init__(self):
        """Initialize HTML converter with default settings."""
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.body_width = 0  # No wrapping

    def html_to_markdown(self, html: str) -> str:
        """
        Convert HTML to markdown format.

        Args:
            html: HTML content to convert

        Returns:
            Markdown representation of the HTML
        """
        if not html:
            return ""
        return self.h2t.handle(html)

    def html_to_clean_text(self, html: str, include_links: bool = False) -> str:
        """
        Convert HTML to clean text.

        Args:
            html: HTML content to convert
            include_links: Whether to include link URLs in output

        Returns:
            Clean text representation without HTML tags
        """
        if not html:
            return ""

        # Create temporary converter with custom settings
        h2t = html2text.HTML2Text()
        h2t.ignore_links = not include_links
        h2t.ignore_images = True
        h2t.body_width = 0

        return h2t.handle(html)
