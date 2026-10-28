"""
Tests for ad/boilerplate pruning and article extraction heuristics.
"""

# Minimal fake HTML mimicking macitynet.it-style markup: ad-widget <article>s
# before the real one (class='maz-banner'), an amazon widget inside the
# article body, and a menu/header.
AD_WIDGET = (
    '<article class="maz-banner">'
    '<div class="maz-banner-title">-52% Samsung MicroSD</div>'
    '<div class="maz-banner-price">67,00 &euro;</div>'
    '<a class="maz-banner-cta">Vedi su Amazon</a>'
    "</article>"
)

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from modulle.web.fetcher import WebFetcher  # noqa: E402

PAGE = f"""
<html><head><title>SwitchBot AI Art Frame 2 review - example</title></head>
<body>
<header><nav><ul><li>Home</li><li>Hi-Tech</li></ul></nav>
<div class="login">Benvenuto! Accedi al tuo account</div></header>
{AD_WIDGET}{AD_WIDGET}{AD_WIDGET}
<article class="post">
  <h1>SwitchBot AI Art Frame 2: caratteristiche, prezzi e novit&agrave;</h1>
  <div class="td-post-content">
    <p>SwitchBot rinnova le sue cornici digitali con AI Art Frame 2: il display
    E-Ink resta il cuore del progetto, con piu memoria e autonomia.</p>
    <div class="asa2_horizontal_box"><div class="asa2_buy_button">MACITYNET +
    Vedi su Amazon</div><div class="maz-banner-price">49,99 &euro;</div></div>
    <p>La seconda generazione mantiene la tecnologia E Ink Spectra 6, uno
    schermo a colori privo di retroilluminazione.</p>
  </div>
</article>
<footer>Privacy Policy - Cookie Policy</footer>
</body></html>
"""


def _fetch_page_mock():
    return {
        "url": "https://example.test/switchbot",
        "title": "SwitchBot AI Art Frame 2 review - example",
        "text": "",
        "html": PAGE,
        "status_code": 200,
        "headers": {},
    }


class TestFetchPagePruning:
    def test_ads_and_chrome_removed_from_text(self):
        fetcher = WebFetcher()
        with patch.object(fetcher, "session") as mock_session:
            mock_session.get.return_value = MagicMock(
                ok=True,
                status_code=200,
                content=PAGE.encode(),
                text=PAGE,
                url="https://example.test/switchbot",
                headers={},
                raise_for_status=lambda: None,
            )
            page = fetcher.fetch_page("https://example.test/switchbot")

        text = page["text"]
        assert "Vedi su Amazon" not in text
        assert "67,00" not in text
        assert "Benvenuto! Accedi al tuo account" not in text
        assert "Privacy Policy" not in text
        # real content survives
        assert "E Ink Spectra 6" in text

    def test_html_dict_still_raw(self):
        fetcher = WebFetcher()
        with patch.object(fetcher, "session") as mock_session:
            mock_session.get.return_value = MagicMock(
                ok=True,
                status_code=200,
                content=PAGE.encode(),
                text=PAGE,
                url="https://example.test/switchbot",
                headers={},
                raise_for_status=lambda: None,
            )
            page = fetcher.fetch_page("https://example.test/switchbot")
        assert page["html"] == PAGE  # raw HTML untouched for markdown path


class TestFetchArticleContent:
    def test_picks_real_article_not_ad_widgets(self):
        fetcher = WebFetcher()
        with patch.object(fetcher, "session") as mock_session:
            mock_session.get.return_value = MagicMock(
                ok=True,
                status_code=200,
                content=PAGE.encode(),
                text=PAGE,
                url="https://example.test/switchbot",
                headers={},
                raise_for_status=lambda: None,
            )
            content = fetcher.fetch_article_content("https://example.test/switchbot")

        assert content
        assert "E Ink Spectra 6" in content
        assert "Vedi su Amazon" not in content
        assert "maz-banner" not in (content or "")

    def test_inline_amazon_widget_still_pruned_in_article(self):
        fetcher = WebFetcher()
        with patch.object(fetcher, "session") as mock_session:
            mock_session.get.return_value = MagicMock(
                ok=True,
                status_code=200,
                content=PAGE.encode(),
                text=PAGE,
                url="https://example.test/switchbot",
                headers={},
                raise_for_status=lambda: None,
            )
            content = fetcher.fetch_article_content("https://example.test/switchbot")
        assert "MACITYNET" not in content
        assert "49,99" not in content
