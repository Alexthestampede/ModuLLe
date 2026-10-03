"""HRR memory module.

Holographic Reduced Representations — pure-python episodic memory with
associative recall that can be injected into LLM context.
"""

from .hrr import (
    HRRMemoryStore,
    text_vector,
    hrr_similarity,
    hrr_convolve,
    hrr_correlate,
)

__all__ = [
    'HRRMemoryStore',
    'text_vector',
    'hrr_similarity',
    'hrr_convolve',
    'hrr_correlate',
]