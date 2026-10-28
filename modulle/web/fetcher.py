"""Web page fetching and parsing utilities."""

import re
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup

from modulle.utils.http_client import create_session, fetch_url
from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)

# Elements removed before text extraction. Includes script/style/nav chrome
# plus the common ad/tracking containers (Google AdSense/GPT, taboola,
# outbrain, amazon affiliate widgets such as macitynet's 'maz-banner').
_STRIP_TAGS = [
    "script",
    "style",
    "noscript",
    "template",
    "iframe",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "ins",
    "svg",
]

# Substring match on id/class: container is dropped entirely. Order matters
# not; match is case-insensitive. Deliberately conservative - these tokens
# virtually never appear in article prose markup.
_AD_CLASS_TOKENS = [
    "adsbygoogle",
    "googlesyndication",
    "doubleclick",
    "taboola",
    "outbrain",
    "advert",
    "advertisement",
    "-ad-",
    "ad-banner",
    "ad-wrapper",
    "adwidget",
    "affiliate",
    "sponsored",
    "amazon-auto-links",
    "asa2_",
    "maz-banner",
    "buy-button",
    "product-box",
    "newsletter-signup",
]

# Exact-match token guards (avoid overmatching e.g. 'badge' via '-ad-')
_AD_EXACT_TOKENS = ["ad", "ads", "adv", "adsense"]

# Site chrome beyond the built-in header/nav/footer/footer tags: hidden
# login/consent popups, mobile menus, publisher disclaimers, breadcrumbs,
# and "related posts" blocks (often <article>-wrapped on WP themes).
# NOTE: keep theme-content containers safe - e.g. 'td-post-content' is the
# main body on WP Newspaper themes, so only 'td-post-next-prev'-style
# tokens are pruned, never a bare 'content'/'post-content' substring.
_CHROME_CLASS_TOKENS = [
    "mfp-hide",
    "mobile-content",
    "td-mobile-",
    "td-login",
    "td-header-",
    "tdc-header-",
    "tdm-descr",
    "cookie-",
    "consent-",
    "gdpr-",
    "breadcrumb",
    "related-posts",
    "relatedposts",
    "jp-related",
    "next-prev",
    "td_module_",
    "td-module-",
    # footer-of-article promos: sidebar widgets, CTA buttons, prev/next,
    # tag lines, share rows (WP Newspaper theme tokens)
    "td_block_widget",
    "tds-call-to-action",
    "tdm_block_button",
    "tdb-next-post",
    "tdb_single_next_prev",
    "td-post-sharing",
    "tdb-tags",
    "tdb_single_tags",
    "wpb_widget_sidebar",
]


def _is_ad_container(tag) -> bool:
    """Heuristic check: is this element an ad/promo container?"""
    tokens = set((tag.get("class") or []))
    if tag.get("id"):
        tokens.add(tag.get("id"))
    lowered = {t.lower() for t in tokens}
    for token in _AD_CLASS_TOKENS:
        if any(token in t for t in lowered):
            return True
    for token in _AD_EXACT_TOKENS:
        if token in lowered:
            return True
    return False


def _is_chrome(tag) -> bool:
    """Heuristic check: is this element site chrome (menus, login/consent
    popups, publisher notices) rather than article content?"""
    tokens = set((tag.get("class") or []))
    if tag.get("id"):
        tokens.add(tag.get("id"))
    lowered = {t.lower() for t in tokens}
    if any(token in t for token in _CHROME_CLASS_TOKENS for t in lowered):
        return True
    # Widgetized list blocks: td_block_<number> (sidebar/promo lists inside
    # WP Newspaper articles). Content blocks never carry a numeric variant.
    return any(re.fullmatch(r"td_block_\d+", t) for t in lowered)


def _prune_soup(soup) -> None:
    """Remove chrome, ads, and tracking elements from a parsed page.

    Mutates the soup in place. Keeps raw-article <article> elements whose
    class names don't look like ad widgets (e.g. WordPress 'maz-banner'
    promo cards), and iterates because nested matches decompose parents.
    """
    for element in soup(_STRIP_TAGS):
        element.decompose()

    # Ad containers by id/class token, innermost-first so nested matches
    # don't double-process decomposed nodes
    removals = [el for el in soup.find_all(True) if _is_ad_container(el) or _is_chrome(el)]
    for el in reversed(removals):
        if getattr(el, "decomposed", False):
            continue
        # Skip if an ancestor was already removed
        if any(a.decomposed for a in el.parents if hasattr(a, "decomposed")):
            continue
        el.decompose()


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
            Dict with keys: url, title, text, html, status_code, headers.
            'text' is cleaned (ads, nav, forms, footers removed) while
            'html' stays the raw response for format='markdown'/'html'.
            Returns None on error
        """
        try:
            response = fetch_url(url, session=self.session, timeout=timeout)

            # Parse with BeautifulSoup using lxml parser (fallback to html.parser)
            soup = self._parse(response.content)

            _prune_soup(soup)

            # Extract title
            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Extract text content
            text = soup.get_text(separator="\n", strip=True)

            return {
                "url": response.url,
                "title": title,
                "text": text,
                "html": response.text,
                "status_code": response.status_code,
                "headers": dict(response.headers),
            }

        except Exception as e:
            logger.error(f"Failed to fetch page {url}: {e}")
            return None

    @staticmethod
    def _parse(content) -> BeautifulSoup:
        """Parse HTML with lxml, falling back to html.parser."""
        try:
            return BeautifulSoup(content, "lxml")
        except Exception:
            return BeautifulSoup(content, "html.parser")

    def clean_html(self, html: str) -> str:
        """Return HTML with ads, chrome, and tracking elements removed.

        Used by the markdown conversion path so LLM-facing formats get the
        same pruning as extract_text. Raw HTML stays available via
        format='html'.
        """
        soup = self._parse(html)
        _prune_soup(soup)
        return str(soup)

    def fetch_article_content(self, url: str) -> Optional[str]:
        """
        Fetch main article content from a page.
        Uses heuristics to extract main content area.

        Args:
            url: URL to fetch

        Returns:
            Extracted article text or None on error
        """
        page_data = self.fetch_page(url)
        if not page_data:
            return None

        try:
            soup = self._parse(page_data["html"])
            _prune_soup(soup)

            # Prefer the largest real <article> over the first one: many
            # sites (e.g. macitynet.it) render ad widgets as <article
            # class="maz-banner"> BEFORE the actual post's <article>, so
            # soup.find('article') grabs an ad and misses the story.
            best_article = None
            best_len = 0
            for article in soup.find_all("article"):
                excerpt = article.get_text(separator="\n", strip=True)
                # A text-length heuristic is ad-resistant: promo cards are
                # tiny; real posts carry paragraphs.
                if len(excerpt) > best_len:
                    best_article = article
                    best_len = len(excerpt)

            if best_article and best_len >= 500:
                # Re-prune inside the article: theme chrome (breadcrumbs,
                # next-prev, author boxes, related grids) often nests inside
                # the <article> element itself.
                inner = [
                    el
                    for el in best_article.find_all(True)
                    if _is_ad_container(el) or _is_chrome(el)
                ]
                for el in reversed(inner):
                    if not getattr(el, "decomposed", False):
                        el.decompose()
                text = best_article.get_text(separator="\n", strip=True)

            if best_article and best_len >= 500 and text:
                return text

            # Try main content area
            main = soup.find("main")
            if main:
                return main.get_text(separator="\n", strip=True)

            # Try content div with common class names
            for class_name in ["content", "article-content", "post-content", "entry-content"]:
                content_div = soup.find("div", class_=class_name)
                if content_div:
                    return content_div.get_text(separator="\n", strip=True)

            # Fallback to full body text
            return page_data["text"]

        except Exception as e:
            logger.error(f"Failed to extract article content from {url}: {e}")
            return None
