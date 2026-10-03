"""File ingestion for ModuLLe.

Turns files (text, source code, PDF, HTML...) into LLM-ready context blocks,
with strategies to avoid blowing up the context window:

- ``inject``: include the (possibly truncated) content inline
- ``summarize``: extract key excerpts
- ``attach``: include name/size only (user can toggle later)

Example:
    >>> from modulle.files import FileIngestor
    >>> ing = FileIngestor()
    >>> block = ing.ingest('notes.md')
    >>> block.to_context_string()
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)


class IngestStrategy(str, Enum):
    """How file content is placed into the LLM context."""
    INJECT = "inject"          # content inline (truncated if huge)
    SUMMARIZE = "summarize"    # key excerpts only
    ATTACH = "attach"          # name + size only


@dataclass
class ContextFile:
    """A file prepared for inclusion in the LLM context."""
    name: str
    path: str
    mime: str
    text: str                    # extracted text (before truncation)
    size_bytes: int
    strategy: IngestStrategy = IngestStrategy.INJECT
    truncated: bool = False
    error: Optional[str] = None
    max_chars: int = 60_000      # per-file inline budget

    def content_for_context(self) -> str:
        """Return the text actually injected, respecting strategy budget."""
        if self.error is not None:
            return f"[unreadable file: {self.error}]"
        if self.strategy == IngestStrategy.ATTACH:
            return f"[file '{self.name}' ({self.size_bytes} bytes) referenced, not included]"
        text = self.text
        if len(text) > self.max_chars:
            cut = text[:self.max_chars]
            # try to cut at a line boundary
            nl = cut.rfind('\n')
            if nl > self.max_chars // 2:
                cut = cut[:nl]
            self.truncated = True
            return cut + f"\n\n[... truncated: showing first {self.max_chars:,} of {len(text):,} chars]"
        return text

    def to_context_string(self) -> str:
        """Format this file as a context block for the LLM."""
        parts = [
            f"<file name=\"{self.name}\" path=\"{self.path}\" "
            f"mime=\"{self.mime}\" size={self.size_bytes}>"
        ]
        if self.error is not None:
            parts.append(f"[error reading file: {self.error}]")
        else:
            if self.strategy == IngestStrategy.ATTACH:
                parts.append(f"[attached reference only — ask user to paste content "
                             f"if needed ({self.size_bytes} bytes)]")
            elif self.strategy == IngestStrategy.SUMMARIZE:
                parts.append("[excerpts mode]")
                parts.append(_key_excerpts(self.content_for_context()))
            else:
                parts.append(self.content_for_context())
        parts.append("</file>")
        return "\n".join(parts)


# Extensions considered literal text -> injected verbatim
_TEXT_EXTS = {
    '.txt', '.md', '.markdown', '.rst', '.log',
    '.py', '.pyw', '.js', '.mjs', '.cjs', '.ts', '.tsx', '.jsx',
    '.c', '.h', '.cpp', '.hpp', '.cc', '.hh', '.rs', '.go', '.java',
    '.kt', '.swift', '.rb', '.php', '.sh', '.bash', '.zsh', '.fish',
    '.lua', '.pl', '.r', '.sql', '.css', '.scss', '.less', '.html',
    '.htm', '.xml', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
    '.conf', '.env', '.csv', '.tsv', '.diff', '.patch', '.vim',
    '.dockerfile', '.gitignore', '.gitattributes', '.service', '.desktop',
}
_BINARY_SKIP = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.ico',
                '.mp3', '.mp4', '.mkv', '.wav', '.zip', '.tar', '.gz', '.bz2',
                '.xz', '.7z', '.rar', '.iso', '.exe', '.dll', '.so', '.dylib',
                '.bin', '.o', '.a', '.pyc', '.db', '.sqlite', '.pdf.image'}
_CODE_BLOCK_LANG = {
    '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
    '.rs': 'rust', '.go': 'go', '.java': 'java', '.sh': 'bash',
    '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml', '.html': 'html',
    '.css': 'css', '.sql': 'sql', '.c': 'c', '.cpp': 'cpp',
}


def _key_excerpts(text: str, block_count: int = 8) -> str:
    """Extract evenly-spaced excerpt blocks from text (cheap 'summarize')."""
    if not text:
        return ''
    block_size = max(1200, len(text) // (block_count * 4) or 1200)
    step = max(1, (len(text) - block_size) // max(1, block_count - 1)) if len(text) > block_size else 1
    excerpts = []
    for i in range(min(block_count, max(1, len(text) // block_size or 1))):
        start = i * step
        chunk = text[start:start + block_size].strip()
        if chunk:
            excerpts.append(chunk)
    seen, out = set(), []
    for e in excerpts:
        key = e[:80]
        if key not in seen:
            seen.add(key)
            out.append(f"[excerpt @ {i}]" if False else e)
    return "\n[...]\n".join(out)


class FileIngestor:
    """Ingest files into LLM-ready context blocks.

    Args:
        default_max_chars: Per-file inline character budget.
        pdf_backend:callable|None — override PDF extraction in tests.
    """

    def __init__(self, default_max_chars: int = 60_000):
        self.default_max_chars = default_max_chars

    # -- public ------------------------------------------------------------
    def ingest(
        self,
        path: str,
        strategy: IngestStrategy = IngestStrategy.INJECT,
        max_chars: Optional[int] = None,
    ) -> ContextFile:
        """Read a file and return a ContextFile ready for context injection."""
        p = Path(path).expanduser()
        max_chars = max_chars if max_chars is not None else self.default_max_chars

        if not p.exists():
            return ContextFile(
                name=p.name, path=str(p), mime='unknown', text='',
                size_bytes=0, strategy=strategy, max_chars=max_chars,
                error=f"file not found: {p}",
            )
        size = p.stat().st_size
        suffix = p.suffix.lower()
        mime = self._guess_mime(p, suffix)

        try:
            if suffix == '.pdf':
                text = self._extract_pdf(p)
            elif suffix in ('.html', '.htm'):
                text = self._extract_html(p)
            elif suffix in _BINARY_SKIP and suffix != '.pdf.image':
                text = ''
                return ContextFile(
                    name=p.name, path=str(p), mime=mime, text='',
                    size_bytes=size, strategy=IngestStrategy.ATTACH, max_chars=max_chars,
                    error=f"binary type '{suffix}' not ingestible; attached as reference",
                )
            else:
                text = self._extract_text(p)
        except Exception as e:
            logger.error(f"Failed to ingest {p}: {e}")
            return ContextFile(
                name=p.name, path=str(p), mime=mime, text='',
                size_bytes=size, strategy=strategy, max_chars=max_chars,
                error=str(e),
            )

        return ContextFile(
            name=p.name, path=str(p), mime=mime, text=text,
            size_bytes=size, strategy=strategy, max_chars=max_chars,
        )

    def ingest_many(self, paths: List[str], **kwargs) -> List[ContextFile]:
        """Ingest multiple files."""
        return [self.ingest(p, **kwargs) for p in paths]

    @staticmethod
    def is_text_file(path: str) -> bool:
        """Cheap check: extension known-text, else sniff for NUL bytes."""
        p = Path(path).expanduser()
        if p.suffix.lower() in _TEXT_EXTS:
            return True
        if p.suffix.lower() == '.pdf':
            return True
        try:
            with open(p, 'rb') as f:
                chunk = f.read(4096)
            return b'\x00' not in chunk
        except OSError:
            return False

    # -- extraction helpers --------------------------------------------------
    @staticmethod
    def _guess_mime(p: Path, suffix: str) -> str:
        import mimetypes
        guess, _ = mimetypes.guess_type(p.name)
        return guess or {
            '.md': 'text/markdown', '.py': 'text/x-python',
        }.get(suffix, 'application/octet-stream')

    @staticmethod
    def _extract_text(p: Path) -> str:
        errors = []
        for enc in ('utf-8', 'utf-16', 'latin-1'):
            try:
                return p.read_text(encoding=enc)
            except (UnicodeDecodeError, UnicodeError) as e:
                errors.append(f"{enc}: {e}")
        # last resort: lossy decode with replacement
        return p.read_bytes().decode('utf-8', errors='replace')

    @staticmethod
    def _extract_pdf(p: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError(
                "PDF support requires the 'pypdf' package "
                "(pip install pypdf)") from e
        reader = PdfReader(str(p))
        parts = []
        for i, page in enumerate(reader.pages, 1):
            try:
                parts.append(page.extract_text() or '')
            except Exception as e:  # per-page robustness
                parts.append(f"[page {i}: extraction error: {e}]")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_html(p: Path) -> str:
        raw = p.read_text(encoding='utf-8', errors='replace')
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw, 'html.parser')
            for t in soup(['script', 'style', 'noscript']):
                t.decompose()
            return soup.get_text(separator='\n', strip=True)
        except ImportError:
            # crude fallback: strip tags
            return re.sub(r'<[^>]+>', ' ', re.sub(r'<(script|style)[^>]*>.*?</\1>',
                                                  '', raw, flags=re.S | re.I))