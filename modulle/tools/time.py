"""Time and date tools for ModuLLe.

Zero-dependency tools that give LLMs access to the current time and date.

Example:
    >>> from modulle.tools.time import CurrentTimeTool, CurrentDateTool
    >>> from modulle.tools import ToolRegistry
    >>>
    >>> registry = ToolRegistry()
    >>> registry.register(CurrentTimeTool())
    >>> registry.register(CurrentDateTool())
"""

from datetime import datetime
from typing import Any, Dict, Optional

from modulle.utils.logging_config import get_logger

from .base import BaseTool

logger = get_logger(__name__)


def _format_offset(dt: datetime) -> str:
    """Return the UTC offset as +HH:MM (local time) or +00:00 (UTC)."""
    offset = dt.utcoffset()
    if offset is None:
        offset = dt.astimezone().utcoffset()
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    return f"{sign}{total_minutes // 60:02d}:{total_minutes % 60:02d}"


class CurrentTimeTool(BaseTool):
    """Tool that provides the current local time. Zero dependencies."""

    def __init__(self, tz: Optional[Any] = None):
        """
        Args:
            tz: Optional timezone (datetime.tzinfo). Defaults to system local time.
                Pass datetime.timezone.utc to always report UTC.
        """
        self.tz = tz

    def get_name(self) -> str:
        return "get_current_time"

    def get_description(self) -> str:
        return (
            "Get the current local time (HH:MM:SS), timezone offset, and "
            "weekday. Use whenever the time of day matters for a task."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs) -> str:
        try:
            now = datetime.now(self.tz) if self.tz else datetime.now()
            stamp = now.strftime("%H:%M:%S") + f" ({_format_offset(now)})"
            text = f"Current time: {stamp}, {now.strftime('%A')}"
            logger.info(text)
            return text
        except Exception as e:
            logger.error(f"CurrentTimeTool failed: {e}")
            return f"Error getting current time: {str(e)}"


class CurrentDateTool(BaseTool):
    """Tool that provides the current date. Zero dependencies."""

    def __init__(self, tz: Optional[Any] = None):
        """
        Args:
            tz: Optional timezone (datetime.tzinfo). Defaults to system local time.
                Pass datetime.timezone.utc to always report UTC.
        """
        self.tz = tz

    def get_name(self) -> str:
        return "get_current_date"

    def get_description(self) -> str:
        return (
            "Get today's date (YYYY-MM-DD) with the weekday name. Use "
            "whenever the date matters for a task."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs) -> str:
        try:
            now = datetime.now(self.tz) if self.tz else datetime.now()
            text = f"Today's date: {now.strftime('%Y-%m-%d')}, {now.strftime('%A')}"
            logger.info(text)
            return text
        except Exception as e:
            logger.error(f"CurrentDateTool failed: {e}")
            return f"Error getting current date: {str(e)}"


class CurrentDateTimeTool(BaseTool):
    """Tool that provides the current date and time in one call. Zero dependencies."""

    def __init__(self, tz: Optional[Any] = None):
        """
        Args:
            tz: Optional timezone (datetime.tzinfo). Defaults to system local time.
                Pass datetime.timezone.utc to always report UTC.
        """
        self.tz = tz

    def get_name(self) -> str:
        return "get_current_datetime"

    def get_description(self) -> str:
        return (
            "Get the current date and time together "
            "(YYYY-MM-DD HH:MM:SS and timezone offset). Use when both matter "
            "or when computing a timestamp."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs) -> str:
        try:
            now = datetime.now(self.tz) if self.tz else datetime.now()
            text = (
                f"Current date and time: "
                f"{now.strftime('%Y-%m-%d %H:%M:%S')} ({_format_offset(now)})"
            )
            logger.info(text)
            return text
        except Exception as e:
            logger.error(f"CurrentDateTimeTool failed: {e}")
            return f"Error getting current date/time: {str(e)}"


__all__ = ["CurrentTimeTool", "CurrentDateTool", "CurrentDateTimeTool"]
