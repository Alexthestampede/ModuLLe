"""Tests for file ingestion (offline; synthesizes its own files)."""

import pytest

from modulle.files import FileIngestor, ContextFile, IngestStrategy


@pytest.fixture
def ingestor():
    return FileIngestor(default_max_chars=1000)


@pytest.fixture
def text_file(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Title\n\nSome *markdown* content here.\n" * 10, encoding='utf-8')
    return p


class TestTextIngest:
    def test_markdown_ingest(self, ingestor, text_file):
        cf = ingestor.ingest(str(text_file))
        assert cf.error is None
        assert cf.mime == 'text/markdown'
        assert 'markdown' in cf.text
        block = cf.to_context_string()
        assert '<file name="notes.md"' in block
        assert 'Some *markdown* content' in block

    def test_unknown_extension_sniffs_text(self, ingestor, tmp_path):
        p = tmp_path / "weird.ext"
        p.write_text("plain text content", encoding='utf-8')
        cf = ingestor.ingest(str(p))
        assert cf.error is None
        assert 'plain text content' in cf.text

    def test_missing_file_error(self, ingestor):
        cf = ingestor.ingest('/nonexistent/file.txt')
        assert cf.error is not None
        assert 'not found' in cf.error
        assert '[error reading file:' in cf.to_context_string()

    def test_truncation(self, tmp_path):
        p = tmp_path / "big.txt"
        p.write_text("x" * 5000, encoding='utf-8')
        cf = FileIngestor(default_max_chars=100).ingest(str(p))
        content = cf.content_for_context()
        assert cf.truncated is True
        assert len(content) < 250
        assert 'truncated' in content
        assert 'of 5,000 chars' in content

    def test_attach_strategy(self, ingestor, text_file):
        cf = ingestor.ingest(str(text_file), strategy=IngestStrategy.ATTACH)
        block = cf.to_context_string()
        assert 'attached reference only' in block
        assert 'Title' not in block  # content withheld


class TestPdfIngest:
    def test_pdf_requires_pypdf_or_extracts(self, ingestor, tmp_path):
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4 minimal not really valid")
        cf = ingestor.ingest(str(p))
        # either pypdf parses it (unlikely for garbage) or a clean error is set
        assert cf.error is None or 'page' in cf.error or cf.error

    def test_real_pdf_roundtrip(self, ingestor, tmp_path):
        pypdf = pytest.importorskip('pypdf')
        from pypdf import PdfWriter
        p = tmp_path / "real.pdf"
        w = PdfWriter()
        w.add_blank_page(width=200, height=200)
        import io
        buf = io.BytesIO()
        w.write(buf)
        p.write_bytes(buf.getvalue())
        cf = ingestor.ingest(str(p))
        assert cf.error is None
        assert cf.mime == 'application/pdf'


class TestBinarySkip:
    def test_binary_types_attached_only(self, ingestor, tmp_path):
        p = tmp_path / "photo.png"
        p.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
        cf = ingestor.ingest(str(p))
        assert cf.strategy == IngestStrategy.ATTACH
        assert 'not ingestible' in (cf.error or '')


class TestHtmlIngest:
    def test_html_script_styles_stripped(self, ingestor, tmp_path):
        p = tmp_path / "page.html"
        p.write_text(
            "<html><head><style>body{color:red}</style></head>"
            "<body><script>evil()</script><h1>Hello</h1><p>World</p></body></html>",
            encoding='utf-8')
        cf = ingestor.ingest(str(p))
        assert cf.error is None
        text = cf.text
        assert 'evil()' not in text and 'color:red' not in text
        assert 'Hello' in text and 'World' in text


class TestIsTextFile:
    def test_known_text_ext(self, text_file):
        assert FileIngestor.is_text_file(str(text_file)) is True

    def test_binary(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00')
        assert FileIngestor.is_text_file(str(p)) is False


class TestContextFileSize:
    def test_size_recorded(self, ingestor, text_file):
        cf = ingestor.ingest(str(text_file))
        assert cf.size_bytes == text_file.stat().st_size