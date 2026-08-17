"""The MCP layer, against fake sessions.

The test that matters most is the isError one: MCP reports most tool failures as a
*successful* response carrying isError: true. If Charter returns that as a value,
BoundFlow records a failure as a success — and on_failure, tool_failures rules, and
every failure metric silently stop working.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from charter.config.loader import load_agent
from charter.mcp.client import Connection, McpError, QuarantineError, ToolSet

EXAMPLES = Path(__file__).parent.parent / "examples"


@dataclass
class FakeTool:
    """MCP 2.0 field names — snake_case. The 1.x camelCase spellings are covered
    separately, since reading the wrong one for is_error would treat every tool
    failure as a success."""
    name: str
    description: str = "does a thing"
    input_schema: dict = field(default_factory=dict)


@dataclass
class FakeContent:
    text: str | None = None
    type: str = "text"


@dataclass
class FakeResult:
    content: list = field(default_factory=list)
    is_error: bool = False


@dataclass
class LegacyResult:
    """An MCP 1.x-shaped result."""
    content: list = field(default_factory=list)
    isError: bool = False


@dataclass
class FakeListing:
    tools: list


class FakeSession:
    def __init__(self, tools: list[FakeTool], results: dict | None = None):
        self._tools = tools
        self._results = results or {}
        self.calls: list[tuple] = []

    async def list_tools(self):
        return FakeListing(self._tools)

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        out = self._results.get(name, FakeResult([FakeContent("ok")]))
        if isinstance(out, Exception):
            raise out
        return out


def run(coro):
    return asyncio.run(coro)


def conn(tools, results=None, spec=None):
    cfg = load_agent(EXAMPLES / "refund-triage").latest
    spec = spec or next(s for s in cfg.mcp if s.name == "stripe")
    return Connection(spec, FakeSession(tools, results))


STRIPE_TOOLS = [
    FakeTool("get_charge"), FakeTool("list_refunds"), FakeTool("create_refund"),
]


class TestDiscovery:
    def test_declared_tools_present(self):
        c = conn(STRIPE_TOOLS)
        run(c.discover())
        assert set(c.available) == {"get_charge", "list_refunds", "create_refund"}

    def test_missing_declared_tool_quarantines(self):
        c = conn([FakeTool("get_charge"), FakeTool("list_refunds")])
        with pytest.raises(QuarantineError, match="does not expose declared tool"):
            run(c.discover())

    def test_undeclared_extra_tools_are_ignored_not_fatal(self):
        """Being strict here would break every agent whenever an upstream server
        ships a release."""
        c = conn(STRIPE_TOOLS + [FakeTool("delete_customer"), FakeTool("charge_card")])
        run(c.discover())
        assert "delete_customer" in c.available  # discovered...
        # ...but never handed to the model — see TestToolSet.


class TestFailureTranslation:
    def test_is_error_raises(self):
        """The whole failure story depends on this one mapping."""
        c = conn(STRIPE_TOOLS, {"get_charge": FakeResult([FakeContent("no such charge")], is_error=True)})
        run(c.discover())
        with pytest.raises(McpError, match="no such charge"):
            run(c.call("get_charge", {}))

    def test_is_error_without_content_still_raises(self):
        c = conn(STRIPE_TOOLS, {"get_charge": FakeResult([], is_error=True)})
        run(c.discover())
        with pytest.raises(McpError, match="reported an error"):
            run(c.call("get_charge", {}))

    def test_transport_failure_raises(self):
        c = conn(STRIPE_TOOLS, {"get_charge": ConnectionResetError("boom")})
        run(c.discover())
        with pytest.raises(McpError, match="ConnectionResetError"):
            run(c.call("get_charge", {}))

    def test_success_returns_text(self):
        c = conn(STRIPE_TOOLS, {"get_charge": FakeResult([FakeContent("amount: 240")])})
        run(c.discover())
        assert run(c.call("get_charge", {})) == "amount: 240"

    def test_content_describing_failure_is_not_detectable(self):
        """A tool that returns ordinary content saying it failed looks like success
        to everyone, us included. Documented, not solved."""
        c = conn(STRIPE_TOOLS, {"get_charge": FakeResult([FakeContent("Error: not found")])})
        run(c.discover())
        assert run(c.call("get_charge", {})) == "Error: not found"  # no raise

    def test_legacy_camelcase_is_error_still_raises(self):
        """If the SDK's field naming moves under us, failures must not silently
        become successes."""
        c = conn(STRIPE_TOOLS, {"get_charge": LegacyResult([FakeContent("gone")], isError=True)})
        run(c.discover())
        with pytest.raises(McpError, match="gone"):
            run(c.call("get_charge", {}))

    def test_non_text_content_keeps_its_type(self):
        c = conn(STRIPE_TOOLS, {"get_charge": FakeResult([FakeContent(None, type="image")])})
        run(c.discover())
        assert run(c.call("get_charge", {})) == "[image]"


class TestSchema:
    def test_required_is_folded_into_descriptions(self):
        """BoundFlow's Tool.input_schema is only the properties map, so `required`
        has nowhere to go — it rides in the description instead of vanishing."""
        c = conn([
            FakeTool("get_charge", input_schema={
                "type": "object",
                "properties": {
                    "charge_id": {"type": "string", "description": "the charge"},
                    "expand": {"type": "boolean"},
                },
                "required": ["charge_id"],
            }),
            FakeTool("list_refunds"), FakeTool("create_refund"),
        ])
        run(c.discover())
        props = c.properties("get_charge")
        assert props["charge_id"]["description"] == "the charge (required)"
        assert "required" not in props["expand"].get("description", "")


class TestToolSet:
    def _connected(self, extra=()):
        cfg = load_agent(EXAMPLES / "refund-triage").latest
        ts = ToolSet()
        for spec in cfg.mcp:
            names = [t.tool for t in spec.tools] + list(extra)
            c = Connection(spec, FakeSession([FakeTool(n) for n in names]))
            run(c.discover())
            ts._connections[spec.name] = c
        return cfg, ts

    def test_gated_tools_are_never_handed_to_the_model(self):
        """create_refund is `approval: always` — the model can't call it because it
        never receives it. That's the safety claim, not a runtime check."""
        cfg, ts = self._connected()
        names = {t.name for t in ts.inline_tools(cfg)}
        assert "stripe.create_refund" not in names
        assert "stripe.get_charge" in names
        assert "zendesk.close_ticket" in names  # approval: never -> inline

    def test_undeclared_tools_are_never_handed_to_the_model(self):
        cfg, ts = self._connected(extra=["delete_customer"])
        names = {t.name for t in ts.inline_tools(cfg)}
        assert not any("delete_customer" in n for n in names)

    def test_tools_are_namespaced(self):
        cfg, ts = self._connected()
        assert all("." in t.name for t in ts.inline_tools(cfg))

    def test_call_gated_invokes_the_approved_tool(self):
        cfg, ts = self._connected()
        assert run(ts.call_gated("stripe.create_refund", {"amount": 240})) == "ok"
        assert ts._connections["stripe"].session.calls == [("create_refund", {"amount": 240})]

    def test_call_gated_refuses_an_ungated_tool(self):
        """execute_act must not become a way to run anything the model names."""
        cfg, ts = self._connected()
        with pytest.raises(McpError, match="not an approval-gated tool"):
            run(ts.call_gated("stripe.get_charge", {}))

    def test_call_gated_refuses_an_undeclared_tool(self):
        cfg, ts = self._connected(extra=["delete_customer"])
        with pytest.raises(McpError, match="not declared"):
            run(ts.call_gated("stripe.delete_customer", {}))

    def test_inline_tool_handler_calls_through(self):
        cfg, ts = self._connected()
        tool = next(t for t in ts.inline_tools(cfg) if t.name == "zendesk.get_ticket")
        assert run(tool.handler({"id": "4821"})) == "ok"
