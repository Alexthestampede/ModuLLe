"""Context window compression for ModuLLe.

Keeps a conversation inside its context budget by summarizing oldest turns
into a compact history note once usage crosses a configurable percentage of
the total window. Pure logic — any provider's processor works.

Example:
    >>> from modulle.context import ContextCompressor
    >>> from modulle import create_ai_client
    >>> client, proc, _ = create_ai_client('ollama', text_model='llama3')
    >>> comp = ContextCompressor(processor=proc, max_tokens=8192,
    ...                          threshold_pct=70)
    >>> messages = comp.maybe_compress(messages)
"""

import re
from typing import List, Dict, Optional, Callable

from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_COMPRESSION_PROMPT = (
    "You compress chat history for an AI assistant's long-term memory. "
    "Produce a concise, factual summary in third person that preserves: "
    "user goals and preferences, decisions made, key facts and data, "
    "open questions, and any instructions to retain. "
    "Omit pleasantries and repetition. Maximum 250 words. "
    "Write it as 'Context so far:' notes."
)

DEFAULT_TRIGGER_MARK = "[earlier conversation summarized below]"


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Good enough for budgeting."""
    if not text:
        return 0
    # words heuristic: min of char/4 and word*1.3 bounds
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """Total estimated tokens of a message list (incl. per-message overhead)."""
    return sum(estimate_tokens(m.get('content', '')) + 4
               for m in messages)


class ContextCompressor:
    """Summarize old turns when a conversation approaches its token budget.

    Attributes:
        processor: Text processor used to generate summaries
            (any object with .generate(prompt=...)).
        max_tokens: Total context window of the target model.
        threshold_pct: 0-100; compress when usage crosses this percentage.
        summary_prompt: Instruction prompt used for summarization.
        keep_recent: Number of most-recent turns always kept verbatim.
        trigger_mark: Human/LLM-visible marker inserted before the summary.
        summarizer: Optional replacement for processor.generate — used in
            tests to avoid hitting a live model.
    """

    def __init__(
        self,
        processor=None,
        max_tokens: int = 8192,
        threshold_pct: int = 70,
        summary_prompt: str = DEFAULT_COMPRESSION_PROMPT,
        keep_recent: int = 6,
        trigger_mark: str = DEFAULT_TRIGGER_MARK,
        summarizer: Optional[Callable[[str], str]] = None,
    ):
        if not 0 < threshold_pct <= 100:
            raise ValueError("threshold_pct must be in (0, 100]")
        if max_tokens < 512:
            raise ValueError("max_tokens must be >= 512")
        self.processor = processor
        self.max_tokens = max_tokens
        self.threshold_pct = threshold_pct
        self.summary_prompt = summary_prompt
        self.keep_recent = max(2, keep_recent)
        self.trigger_mark = trigger_mark
        self.summarizer = summarizer

    # -- public ---------------------------------------------------------------
    def usage_pct(self, messages: List[Dict[str, str]]) -> float:
        """Estimated context usage as a percentage of the window."""
        return estimate_messages_tokens(messages) / self.max_tokens * 100.0

    def should_compress(self, messages: List[Dict[str, str]]) -> bool:
        """True if estimated usage crossed the threshold."""
        return self.usage_pct(messages) >= self.threshold_pct

    def maybe_compress(
        self,
        messages: List[Dict[str, str]],
        force: bool = False,
    ) -> List[Dict[str, str]]:
        """Return messages, compressed if the threshold was crossed.

        Compression replaces all messages between the first system message
        and the last ``keep_recent`` messages with a single summary message
        of role ``assistant`` content ``trigger_mark + summary``.

        Returns the original list unchanged when below threshold or when
        there is not enough history to compress (<= keep_recent + 2).
        """
        est = self.usage_pct(messages)
        if not force and est < self.threshold_pct:
            return messages
        if len(messages) <= self.keep_recent + 2:
            return messages
        return self.compress(messages)

    def compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Compress now, ignoring thresholds."""
        est = self.usage_pct(messages)
        head, compressible, tail = self._split(messages)
        if not compressible:
            return messages

        summary = self._summarize(compressible)
        if not summary:
            return messages

        compressed_note = {
            'role': 'assistant',
            'content': f"{self.trigger_mark}\n{summary}",
        }
        result = head + [compressed_note] + tail
        new_est = self.usage_pct(result)
        logger.info(
            f"Context compressed: {len(messages)} -> {len(result)} messages, "
            f"~{est:.0f}% -> ~{new_est:.0f}% of window")
        return result

    # -- internals --------------------------------------------------------------
    @staticmethod
    def _split(messages):
        """Split into (head_system, compressible, tail_recent)."""
        head = []
        idx = 0
        if messages and messages[0].get('role') == 'system':
            head = [messages[0]]
            idx = 1
        tail_start = max(idx, len(messages) - 6)
        return head, messages[idx:tail_start], messages[tail_start:]

    def _summarize(self, compressible) -> Optional[str]:
        transcript = []
        for m in compressible:
            role = m.get('role', 'user').upper()
            content = m.get('content', '')
            transcript.append(f"{role}: {content}")
        text = "\n".join(transcript)
        prompt = f"{self.summary_prompt}\n\nConversation to compress:\n{text}"

        if self.summarizer is not None:
            summary = self.summarizer(prompt)
        elif self.processor is not None:
            summary = self.processor.generate(prompt=prompt, temperature=0.2)
        else:
            logger.warning("No processor/summarizer available; skipping compression")
            return None
        summary = (summary or '').strip()
        return summary or None