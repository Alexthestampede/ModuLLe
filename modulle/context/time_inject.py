"""Time/date context injection for ModuLLe.

Passively prepends a current-date/time line to a conversation's system
prompt so LLMs can answer "what time is it" style questions without tool
round-trips. Complements the active tool-based approach in
``modulle.tools.time``.
"""

from datetime import datetime
from typing import Dict, List, Optional

from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_TIME_TEMPLATE = "Current date and time: {now}."


def format_now(
    now: Optional[datetime] = None,
    template: str = DEFAULT_TIME_TEMPLATE,
) -> str:
    """Format the current time line (injected into the system prompt)."""
    dt = now or datetime.now().astimezone()
    return template.format(now=dt.strftime("%A, %Y-%m-%d %H:%M (%Z)"))


def inject_time(
    messages: List[Dict[str, str]],
    now: Optional[datetime] = None,
    template: str = DEFAULT_TIME_TEMPLATE,
) -> List[Dict[str, str]]:
    """Return messages with a time line merged into the system prompt.

    - Inserts/updates the system message at position 0.
    - Idempotent: replaces any previous ``Current date and time:`` line.
    - Empty message list: returns a new list with just the system entry.
    """
    time_line = format_now(now=now, template=template)
    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content", "")
        # strip any previous injected line
        cleaned = "\n".join(
            ln for ln in content.split("\n") if not ln.startswith("Current date and time:")
        ).strip()
        merged = f"{time_line}\n{cleaned}" if cleaned else time_line
        messages[0] = {"role": "system", "content": merged}
    else:
        messages.insert(0, {"role": "system", "content": time_line})
    return messages
