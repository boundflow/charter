"""Against a real MCP server process — real transport, real discovery, real errors.

This file has earned its keep twice. Fakes written against MCP 1.x field names
passed happily while the installed SDK used 2.0's, so a failing tool looked like a
successful one; and dotted tool names looked fine everywhere except the provider,
which rejects them. Both only showed up here.

It matters more now, not less. Charter no longer speaks MCP — `langchain-mcp-adapters`
does — so what these tests actually check is that our two remaining
responsibilities survive somebody else's client: the declaration, and the ratchet.
"""

import asyncio
import sys
from pathlib import Path

import pytest

from charter.config.agent import ApprovalRules, McpServer, ToolSpec
from charter.mcp.client import QuarantineError, ToolSet

SERVER = str(Path(__file__).parent / "mcp_fixture_server.py")


def spec(tools, approval=None):
    return McpServer(name="tickets", command=sys.executable, args=[SERVER],
                     approval=approval, tools=tools)


class Cfg:
    """Only `.mcp` is read here, so a whole AgentConfig would be noise."""

    def __init__(self, server):
        self.mcp = [server]


DECLARED = [
    ToolSpec(tool="get_ticket"),
    ToolSpec(tool="close_ticket", approval="always", on_failure="fail"),
]


async def _connected(tools=None, approval=None):
    ts = ToolSet()
    await ts.connect(Cfg(spec(tools or DECLARED, approval)))
    return ts


def run(coro):
    return asyncio.run(coro)


def test_connects_and_discovers():
    ts = run(_connected())
    assert set(ts.servers["tickets"].tools) == {"get_ticket", "close_ticket"}


def test_undeclared_tools_never_reach_the_model():
    """The server offers `undeclared_danger`. Nothing declares it, so the model is
    never handed it — that omission is the authority claim, and it has to survive
    the adapter loading every tool the server offers."""
    ts = run(_connected())
    names = [t.name for t in ts.langchain_tools()]
    assert names == ["tickets__get_ticket", "tickets__close_ticket"]
    assert not any("undeclared" in n for n in names)


def test_wire_names_are_double_underscored():
    """A dot is rejected by the provider, and the adapter's own prefixing joins with
    a single underscore, which is ambiguous: server `a_b` tool `c` collides with
    server `a` tool `b_c`. So the rename stays ours."""
    ts = run(_connected())
    for tool in ts.langchain_tools():
        assert "." not in tool.name
        assert tool.name.count("__") == 1


def test_real_schema_survives_the_adapter():
    """The model needs the server's argument schema, not a placeholder."""
    ts = run(_connected())
    tool = next(t for t in ts.langchain_tools() if t.name.endswith("get_ticket"))
    assert "ticket_id" in (tool.args_schema or {}).get("properties", {})


def test_a_real_tool_call_round_trips():
    async def go():
        ts = await _connected()
        tool = next(t for t in ts.langchain_tools() if t.name.endswith("get_ticket"))
        return await tool.ainvoke({"ticket_id": "42"})

    assert "customer wants a refund" in str(run(go()))


def test_a_failing_tool_reports_as_a_failure():
    """The bug this file exists for: a raising tool must not look successful. The
    adapter surfaces it as error content rather than an exception, so what matters
    is that the failure is legible, not how it arrives."""
    async def go():
        ts = await _connected()
        tool = next(t for t in ts.langchain_tools() if t.name.endswith("close_ticket"))
        try:
            return str(await tool.ainvoke({"ticket_id": "7"}))
        except Exception as e:
            return str(e)

    assert "already closed" in run(go())


def test_missing_declared_tool_quarantines():
    """Declaring something the server doesn't have fails at boot, where an operator
    sees it, rather than a round into the first task."""
    with pytest.raises(QuarantineError, match="does not expose declared tool"):
        run(_connected([ToolSpec(tool="refund_everything")]))


def test_annotations_ratchet_against_a_real_server():
    """The fixture marks `get_ticket` read-only and `close_ticket` destructive, so a
    server can tighten a gate we didn't ask for — and never loosen one."""
    ts = run(_connected(
        tools=[ToolSpec(tool="get_ticket"), ToolSpec(tool="close_ticket")],
        approval=ApprovalRules(read_only="never", default="always")))

    assert ts.servers["tickets"].tightened == {"close_ticket": "destructive_hint"}
    assert ts.gated_tools() == ["tickets__close_ticket"]


def test_an_explicit_never_outranks_a_destructive_hint():
    """Config wins. A server can make an agent safer without a deploy; it cannot
    override a decision someone wrote down."""
    ts = run(_connected(
        tools=[ToolSpec(tool="get_ticket"), ToolSpec(tool="close_ticket", approval="never")],
        approval=ApprovalRules(read_only="never", default="always")))

    assert "close_ticket" not in ts.servers["tickets"].tightened
    assert ts.gated_tools() == []


def test_a_failing_tool_reports_its_error_to_the_model():
    """The adapter returns the failure rather than raising it, which is what keeps
    one broken tool from killing a run — the model reads the error and can work
    around it.

    The cost is that BoundFlow's wrapper counts a failure only on a raise, so this
    is currently invisible to tool_failure_counts. See the note in client.py; the
    fix belongs in the wrapper, not here.
    """
    async def go():
        ts = await _connected([ToolSpec(tool="get_ticket"),
                               ToolSpec(tool="close_ticket")])
        tool = next(t for t in ts.langchain_tools() if t.name.endswith("close_ticket"))
        return str(await tool.ainvoke({"ticket_id": "7"}))

    assert "already closed" in run(go())


def test_a_hanging_tool_becomes_a_failure_rather_than_a_dead_round():
    """The adapter offers a timeout for HTTP connections and none for stdio, so a
    hung stdio server blocked until the control plane cancelled the whole
    operation. Bounded, but by the bluntest instrument there is.

    Returned rather than raised, and in the shape a tool that fails without raising
    returns — so it is counted and classified down the same path as any other
    failure, and the agent's declared `on_failure` is what decides. Raising took
    that decision away: it ended the run wherever it stood, so a customer got
    `on_failure: continue` honoured when the tool errored and ignored when it hung.
    """
    async def go():
        ts = ToolSet().with_timeout(0.001)   # nothing answers this fast
        await ts.connect(Cfg(spec([ToolSpec(tool="get_ticket")])))
        tool = ts.langchain_tools()[0]
        return await tool.ainvoke({"ticket_id": "42"})

    answered = str(run(go()))
    assert "no response within" in answered
    # The wording a returned failure is recognised by. Without it the timeout is
    # read as an ordinary result and the agent carries on believing it worked.
    assert "error executing tool" in answered.lower()


def test_no_timeout_configured_leaves_the_tool_alone():
    async def go():
        ts = ToolSet()   # no with_timeout
        await ts.connect(Cfg(spec([ToolSpec(tool="get_ticket")])))
        return str(await ts.langchain_tools()[0].ainvoke({"ticket_id": "42"}))

    assert "customer wants a refund" in run(go())


def test_a_server_that_cannot_start_says_why_not_just_that_it_failed():
    """The transport reports `Connection closed`, because from its side that is all
    that happened — the process died before speaking MCP. Whatever explains it went
    to the subprocess's stderr, which the adapter hands to the worker's own stderr
    and nobody keeps.

    Found the hard way: a task failed with "mcp server 'net': unhandled errors in a
    TaskGroup (1 sub-exception)", and the real cause — a `python` on PATH without
    `mcp` installed — appeared nowhere an operator would look.
    """
    from charter.mcp.client import _startup
    from charter.config.agent import McpServer, ToolSpec

    spec = McpServer(name="broken_one", command=sys.executable,
                     args=["-c", "import nonexistent_module_xyz"],
                     tools=[ToolSpec(tool="t")])

    found = run(_startup(spec, seconds=10))

    assert found.code == "startup_failed"
    assert "nonexistent_module_xyz" in found.verbatim, (
        "the process's own words, not a paraphrase")
    assert found.hint, "a code and a cause without a next step is half a message"


def test_a_missing_command_is_told_apart_from_a_crashing_one():
    """Structural, not by reading stderr. Which failure it was is knowable from how
    it failed — exec refused, versus a process that ran and died — and guessing
    from output text is what goes quietly wrong when wording changes."""
    from charter.mcp.client import _startup
    from charter.config.agent import McpServer, ToolSpec

    spec = McpServer(name="absent_one", command="charter-no-such-binary",
                     args=[], tools=[ToolSpec(tool="t")])

    found = run(_startup(spec, seconds=5))

    assert found.code == "command_not_found"
    assert "PATH" in found.hint, "the fix is almost always PATH or cwd"


def test_a_task_group_error_is_flattened_to_its_causes():
    """`str()` on an ExceptionGroup is the same sentence whatever went wrong."""
    from charter.mcp.client import _leaves

    inner = ValueError("the actual problem")
    try:
        raise ExceptionGroup("unhandled errors in a TaskGroup", [inner])
    except ExceptionGroup as e:
        leaves = _leaves(e)

    assert leaves == ["ValueError: the actual problem"]
