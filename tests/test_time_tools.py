"""
Tests for time/date tools.
"""

from datetime import datetime, timezone

from modulle.tools import ToolRegistry
from modulle.tools.time import (
    CurrentDateTimeTool,
    CurrentDateTool,
    CurrentTimeTool,
    _format_offset,
)


class TestCurrentTimeTool:
    def test_returns_parseable_time(self):
        result = CurrentTimeTool().execute()
        assert result.startswith("Current time: ")
        stamp = result.split(": ", 1)[1]
        datetime.strptime(stamp.split(" (")[0], "%H:%M:%S")
        assert " (" in stamp and "), " in stamp

    def test_utc_explicit(self):
        before = datetime.now(timezone.utc).replace(microsecond=0)
        result = CurrentTimeTool(tz=timezone.utc).execute()
        after = datetime.now(timezone.utc)
        stamp = result.split("Current time: ")[1].split(" (")[0]
        parsed = datetime.strptime(stamp, "%H:%M:%S").replace(
            tzinfo=timezone.utc, year=before.year, month=before.month, day=before.day
        )
        assert before <= parsed <= after
        assert "(+00:00)" in result

    def test_schema(self):
        tool = CurrentTimeTool()
        assert tool.get_name() == "get_current_time"
        schema = tool.to_ollama_schema()
        assert schema["function"]["name"] == "get_current_time"
        assert schema["function"]["parameters"]["type"] == "object"


class TestCurrentDateTool:
    def test_returns_today(self):
        result = CurrentDateTool().execute()
        today = datetime.now().strftime("%Y-%m-%d")
        assert result == f"Today's date: {today}, {datetime.now().strftime('%A')}"


class TestCurrentDateTimeTool:
    def test_returns_both(self):
        result = CurrentDateTimeTool(tz=timezone.utc).execute()
        assert result.startswith("Current date and time: ")
        assert "(+00:00)" in result
        stamp = result.split("Current date and time: ")[1].split(" (")[0]
        datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")


class TestRegistryIntegration:
    def test_register_and_execute_via_registry(self):
        registry = ToolRegistry()
        registry.register(CurrentTimeTool())
        registry.register(CurrentDateTool())
        registry.register(CurrentDateTimeTool())
        assert set(registry.list_tools()) == {
            "get_current_time",
            "get_current_date",
            "get_current_datetime",
        }
        assert registry.execute("get_current_date").startswith("Today's date: ")
        formats = [
            registry.to_ollama_format(),
            registry.to_openai_format(),
            registry.to_claude_format(),
            registry.to_gemini_format(),
        ]
        assert all(len(f) == 3 for f in formats)


def test_format_offset_handles_half_hour_zones():
    from datetime import timedelta, tzinfo

    class Katmandu(tzinfo):
        def utcoffset(self, _dt):
            return timedelta(hours=5, minutes=45)

        def dst(self, _dt):
            return timedelta(0)

    assert _format_offset(datetime(2000, 1, 1, tzinfo=Katmandu())) == "+05:45"
    assert _format_offset(datetime(2000, 1, 1, tzinfo=timezone.utc)) == "+00:00"
