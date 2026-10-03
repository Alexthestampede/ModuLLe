"""Context management utilities.

Currently: context window compression (summarize old turns when the
conversation approaches the model's token budget).
"""

from .compression import (
    ContextCompressor,
    estimate_tokens,
    estimate_messages_tokens,
    DEFAULT_COMPRESSION_PROMPT,
    DEFAULT_TRIGGER_MARK,
)
from .time_inject import inject_time, format_now, DEFAULT_TIME_TEMPLATE

__all__ = [
    'ContextCompressor',
    'estimate_tokens',
    'estimate_messages_tokens',
    'DEFAULT_COMPRESSION_PROMPT',
    'DEFAULT_TRIGGER_MARK',
    'inject_time',
    'format_now',
    'DEFAULT_TIME_TEMPLATE',
]