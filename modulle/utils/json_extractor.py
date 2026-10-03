"""
JSON response extraction utilities.

Handles LLM responses that should contain JSON, tolerating:
- Markdown code fences (```json ... ```)
- Reasoning traces before/after the JSON
- Chatty preambles ("Here is the JSON you requested:")
- Trailing commentary after the closing brace
"""
import json
import re
from typing import Optional, Dict, Any

from .response_cleaner import clean_response

# fenced code block, optionally labeled json
_FENCED_JSON_RE = re.compile(
    r"```(?:json|JSON)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)


def _try_parse(candidate: str) -> Optional[Dict[str, Any]]:
    """Parse a candidate string as a JSON object, return None on failure."""
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def extract_json(text) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from a model response.

    Strategy:
    1. Clean reasoning traces via clean_response()
    2. Try the whole response as JSON
    3. Try fenced ```json blocks
    4. Brace-balance scan for the first complete {...} object

    Args:
        text: Raw model response

    Returns:
        Parsed dict, or None if no JSON object found
    """
    if not text:
        return None

    cleaned = clean_response(text)
    if not cleaned:
        return None

    # 1. Whole response is JSON
    parsed = _try_parse(cleaned)
    if parsed is not None:
        return parsed

    # 2. Fenced JSON block
    match = _FENCED_JSON_RE.search(cleaned)
    if match:
        parsed = _try_parse(match.group(1))
        if parsed is not None:
            return parsed

    # 3. Brace-balance scan: first { to its matching }
    #    (tolerates preambles, trailing commentary, and braces inside strings)
    pos = cleaned.find("{")
    while pos != -1 and pos < len(cleaned):
        start = cleaned.find("{", pos)
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False
        end = -1
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            # Unbalanced opener: the object never closes, but a complete
            # object may start inside it (e.g. "{broken json\n{...}").
            # Retry from the next { after this opener; give up when none.
            next_pos = cleaned.find("{", start + 1)
            if next_pos == -1:
                return None
            pos = next_pos
            continue

        parsed = _try_parse(cleaned[start : end + 1])
        if parsed is not None:
            return parsed

        # malformed object; retry from the next { AFTER this object's opener
        # (a broken opener may have swallowed a valid object that followed)
        pos = start + 1

    return None