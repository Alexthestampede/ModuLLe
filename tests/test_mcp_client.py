"""Tests for the MCP client adapter.

Runs a real local MCP server subprocess (a tiny stdio server written with
the installed `mcp` package) when available; falls back to interface-level
tests that do not need a subprocess.
"""

import json
import subprocess
import sys
import textwrap
import time

import pytest

mcp = pytest.importorskip('mcp')

from modulle.tools.mcp_client import MCPClientManager, MCPTool  # noqa: E402
from modulle.tools import ToolRegistry  # noqa: E402

ECHO_SERVER = textwrap.dedent('''
    import anyio
    import mcp_types as t
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    async def list_tools(ctx, params):
        return t.ListToolsResult(tools=[t.Tool(
            name="echo",
            description="Echo back the given text",
            input_schema={"type": "object",
                         "properties": {"text": {"type": "string"}},
                         "required": ["text"]},
        )])

    async def call_tool(ctx, params):
        name = params.name
        args = params.arguments or {}
        if name != "echo":
            return t.CallToolResult(
                content=[t.TextContent(type="text", text=f"unknown tool {name}")],
                is_error=True)
        return t.CallToolResult(
            content=[t.TextContent(type="text", text=f"echo: {args.get('text', '')}")])

    app = Server("echo-server", on_list_tools=list_tools, on_call_tool=call_tool)

    async def main():
        async with stdio_server() as (r, w):
            await app.run(r, w, app.create_initialization_options())

    if __name__ == "__main__":
        anyio.run(main)
''')


@pytest.fixture(scope='module')
def manager(tmp_path_factory):
    server_py = tmp_path_factory.mktemp('mcp') / 'echo_server.py'
    server_py.write_text(ECHO_SERVER)
    mgr = MCPClientManager()
    failures = mgr.connect_all([{
        'name': 'test', 'transport': 'stdio',
        'command': sys.executable, 'args': [str(server_py)],
    }])
    assert failures == {}
    yield mgr
    mgr.close_all()


class TestConnection:
    def test_connected(self, manager):
        assert manager.is_connected('test')
        assert 'test' in manager.list_servers()

    def test_list_tools(self, manager):
        tools = manager.list_tools('test')
        assert ('test', 'echo', 'Echo back the given text') in tools


class TestRegistryIntegration:
    def test_build_registry_namespaces_tools(self, manager):
        reg = manager.build_registry(server='test')
        assert 'mcp_test_echo' in reg
        assert len(reg) == 1

    def test_tool_schemas(self, manager):
        reg = manager.build_registry()
        tool = reg.get_tool('mcp_test_echo')
        assert isinstance(tool, MCPTool)
        assert '[MCP:test]' in tool.get_description()
        schema = tool.get_parameters()
        assert schema['properties']['text']['type'] == 'string'

    def test_import_into_existing_registry(self, manager):
        reg = ToolRegistry()
        manager.import_registry_into(reg, server='test')
        assert 'mcp_test_echo' in reg


class TestExecution:
    def test_call_tool_roundtrip(self, manager):
        out = manager.call_tool_sync('test', 'echo', {'text': 'hello mcp'})
        assert out == 'echo: hello mcp'

    def test_registry_execute(self, manager):
        reg = manager.build_registry()
        out = reg.execute('mcp_test_echo', text='via registry')
        assert 'echo: via registry' in out

    def test_unknown_server(self, manager):
        out = manager.call_tool_sync('ghost', 'echo', {})
        assert 'not connected' in out

    def test_tool_error_surfaces(self, manager):
        out = manager.call_tool_sync('test', 'echo', {'text': 'x'})
        assert 'echo: x' in out
        # call a nonexistent tool on a real server -> error string raised
        with pytest.raises(RuntimeError):
            manager.call_tool_sync('test', 'no_such_tool', {})


class TestMissing:
    def test_tool_get_name_namespaced(self):
        class FakeMgr:
            def call_tool_sync(self, s, t, a):
                return 'ok'
        tool = MCPTool(FakeMgr(), 'srv', 'ping', 'Pings', {'type': 'object'})
        assert tool.get_name() == 'mcp_srv_ping'