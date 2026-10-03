"""MCP (Model Context Protocol) client adapter for ModuLLe.

Connects to MCP servers (local subprocess or HTTP/SSE endpoints), lists
their tools, and exposes them as ModuLLe ``BaseTool`` objects inside a
standard ``ToolRegistry`` so every provider's tool-calling works unchanged.

Requires the optional ``mcp`` package:  pip install modulle[mcp]  (or simply
``pip install mcp``).

Example:
    >>> import anyio
    >>> from modulle.tools.mcp_client import MCPClientManager
    >>>
    >>> manager = MCPClientManager()
    >>> anyio.run(manager.connect_all, [
    ...     {'name': 'fs', 'transport': 'stdio',
    ...      'command': 'npx', 'args': ['-y', '@modelcontextprotocol/server-filesystem', '/tmp']},
    ...     {'name': 'web', 'transport': 'http', 'url': 'http://localhost:8931/mcp'},
    ... ])
    >>> registry = manager.build_registry(server='fs')   # or None for all
    >>> result = anyio.run(manager.call_tool, 'fs', 'list_directory', {'path': '/tmp'})
"""

import asyncio
import json
import shutil
from typing import Any, Dict, List, Optional

from modulle.tools.base import BaseTool
from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.types import TextContent

    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without extra
    MCP_AVAILABLE = False


class MCPNotAvailable(RuntimeError):
    """Raised when the `mcp` package is not installed."""

    def __init__(self):
        super().__init__("MCP support requires the 'mcp' extra: pip install modulle[mcp]")


class MCPTool(BaseTool):
    """Wraps one remote MCP tool as a ModuLLe BaseTool."""

    def __init__(
        self,
        manager: "MCPClientManager",
        server_name: str,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
    ):
        self._manager = manager
        self._server = server_name
        self._name = name
        self._description = description
        self._schema = input_schema or {"type": "object", "properties": {}}

    # namespaced so tools from different servers cannot collide
    def get_name(self) -> str:
        return f"mcp_{self._server}_{self._name}"

    def get_description(self) -> str:
        server_part = f"[MCP:{self._server}] "
        return f"{server_part}{self._description}".strip()

    def get_parameters(self) -> Dict[str, Any]:
        return self._schema

    def execute(self, **kwargs) -> str:
        """Blocking execute — runs the async call on a private event loop."""
        result = self._manager.call_tool_sync(self._server, self._name, kwargs)
        return result


class MCPClientManager:
    """Manages connections to one or more MCP servers.

    Usage pattern: define servers in config, call ``connect_all`` once at
    startup (or lazily per server), then ``build_registry()`` to merge MCP
    tools into the app's ToolRegistry. Sessions stay open for the process
    lifetime; on GUI shutdown call ``close_all``.
    """

    def __init__(self):
        if not MCP_AVAILABLE:
            raise MCPNotAvailable()
        # server_name -> {'server': ClientSession, 'tools': [(name, desc, schema)], 'spec': {...}}
        self._sessions: Dict[str, Dict[str, Any]] = {}

    # -- connection -------------------------------------------------------------
    def connect(self, spec: Dict[str, Any]):
        """Connect to a single MCP server (blocking).

        spec keys:
          - name: unique label
          - transport: 'stdio' | 'http'
          - stdio: command (str), args (list), env (dict, optional),
                   cwd (str, optional)
          - http: url (str), headers (dict, optional)
        """
        name = spec["name"]
        if name in self._sessions:
            logger.warning(f"MCP server '{name}' already connected")
            return

        async def _run():
            if spec.get("transport") == "http":
                params = {}
                url = spec["url"]
                cm = streamable_http_client(url)
                transport_streams = await cm.__aenter__()
                read_stream, write_stream = transport_streams[0], transport_streams[1]
            else:
                command = spec["command"]
                resolved = shutil.which(command) or command
                args = list(spec.get("args", []))
                params = StdioServerParameters(
                    command=resolved,
                    args=args,
                    env=spec.get("env") or None,
                    cwd=spec.get("cwd") or None,
                )
                cm = stdio_client(params)
                read_stream, write_stream = await cm.__aenter__()
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()
            tools_resp = await session.list_tools()
            tools = [(t.name, t.description or "", t.input_schema or {}) for t in tools_resp.tools]
            return cm, session, tools

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            cm, session, tools = loop.run_until_complete(_run())
        except Exception:
            loop.close()
            raise
        self._sessions[name] = {
            "loop": loop,
            "ctx": cm,
            "session": session,
            "tools": tools,
            "spec": spec,
        }
        logger.info(f"Connected to MCP server '{name}' with tools: " f"{[t[0] for t in tools]}")

    def connect_all(self, specs: List[Dict[str, Any]], continue_on_error: bool = True):
        """Connect to many servers; collects failures when continue_on_error."""
        failures = {}
        for spec in specs:
            try:
                self.connect(spec)
            except Exception as e:
                logger.error(f"MCP connect failed for '{spec.get('name')}': {e}")
                if continue_on_error:
                    failures[spec.get("name", "?")] = str(e)
                else:
                    raise
        return failures

    def disconnect(self, name: str):
        """Close one server session."""
        info = self._sessions.pop(name, None)
        if not info:
            return

        async def _close():
            try:
                await info["session"].__aexit__(None, None, None)
            finally:
                await info["ctx"].__aexit__(None, None, None)

        try:
            asyncio.set_event_loop(info["loop"])
            info["loop"].run_until_complete(_close())
        except Exception as e:
            logger.warning(f"Error closing MCP '{name}': {e}")
        finally:
            info["loop"].close()

    def close_all(self):
        for name in list(self._sessions):
            self.disconnect(name)

    # -- queries -------------------------------------------------------------
    def list_servers(self) -> List[str]:
        return list(self._sessions)

    def list_tools(self, server: Optional[str] = None):
        """List (server, name, description) triples."""
        out = []
        for sname, info in self._sessions.items():
            if server is not None and sname != server:
                continue
            for name, desc, _schema in info["tools"]:
                out.append((sname, name, desc))
        return out

    def is_connected(self, name: str) -> bool:
        return name in self._sessions

    # -- registry ----------------------------------------------------------------
    def build_registry(self, server: Optional[str] = None):
        """Create a ToolRegistry populated with MCP tools (optionally one server).

        Returns a fresh ``ToolRegistry`` — merge with the app's own by
        registering each tool into an existing registry via
        ``import_registry_into(existing)``.
        """
        from modulle.tools import ToolRegistry

        reg = ToolRegistry()
        self.import_registry_into(reg, server=server)
        return reg

    def import_registry_into(self, registry, server: Optional[str] = None):
        """Register MCP tools into an existing ToolRegistry."""
        for sname, info in self._sessions.items():
            if server is not None and sname != server:
                continue
            for name, desc, schema in info["tools"]:
                registry.register(MCPTool(self, sname, name, desc, schema))

    # -- execution -------------------------------------------------------------
    def call_tool_sync(
        self, server: str, tool: str, arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        """Blocking wrapper used by MCPTool.execute (new private loop).

        MCP stdio/http sessions created on another loop cannot be reused from
        a different one, so execution uses a short-lived event loop bound to
        the same session object via anyio.from_thread is not portable here.
        Instead we rely on the session being loop-agnostic for HTTP, and for
        stdio we re-use the connection loop with run_coroutine_threadsafe.
        """
        info = self._sessions.get(server)
        if not info:
            return f"Error: MCP server '{server}' not connected"

        async def _call():
            result = await info["session"].call_tool(tool, arguments or {})
            parts = []
            for block in result.content:
                if isinstance(block, TextContent):
                    parts.append(block.text)
                else:
                    parts.append(json.dumps(block.model_dump()))
            text = "\n".join(parts)
            if getattr(result, "is_error", False):
                raise RuntimeError(text or "MCP tool error")
            return text

        loop = info["loop"]
        try:
            running = loop.is_running()
        except RuntimeError:
            running = False
        if running:
            import concurrent.futures

            fut = asyncio.run_coroutine_threadsafe(_call(), loop)
            try:
                return fut.result(timeout=120)
            except concurrent.futures.TimeoutError:
                fut.cancel()
                return f"Error: MCP tool '{tool}' timed out"
        else:
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_call())
            finally:
                asyncio.set_event_loop(None)
