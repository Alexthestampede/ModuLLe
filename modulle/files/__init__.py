"""File ingestion module — turn files into LLM-ready context blocks.

Example:
    >>> from modulle.files import FileIngestor
    >>> ing = FileIngestor()
    >>> block = ing.ingest('report.pdf')
    >>> prompt_addition = block.to_context_string()
"""

from .ingest import ContextFile, FileIngestor, IngestStrategy

__all__ = ["FileIngestor", "ContextFile", "IngestStrategy"]
