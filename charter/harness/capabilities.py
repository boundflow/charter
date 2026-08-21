"""Name what a tool *does*, so a cap survives the harness shipping a second way to do it.

Capping `write_file` at one works, and then the agent uses `edit_file` to achieve the same
thing. That isn't a broken cap — it's what naming tools instead of capabilities buys you,
and it gets worse as harnesses grow: deepagents ships three separate tools that mutate the
filesystem, and adds more between releases.

deepagents already solved this for its own filesystem, and the convention here is theirs,
deliberately. `FilesystemOperation` is a closed vocabulary — `"read"` and `"write"` — and
`_DEFAULT_FS_TOOL_OPS` maps each filesystem tool to one of them, so a `FilesystemPermission`
is written over operations and paths and never over tool names. Our names and groupings
match theirs exactly, which means

    CapabilityCallLimit(capability="write", max_calls=5)

covers the same three tools that `FilesystemPermission(operations=["write"], ...)` does. A
customer who has read deepagents' docs already knows what `write` means, and the two
mechanisms can't drift into disagreeing about it. What BoundFlow adds isn't a vocabulary,
it's that this one is declarative and versioned — the same rules, written down where they
roll back with the workflow instead of living in whatever code built the agent.

Where deepagents stops, we continue. Its map is private, filesystem-only, and not an
extension point, so `execute` and `task` — the two most consequential tools it ships — have
no operation at all. Those we name ourselves, and customers can name their own with
`register_capability`.

**Not a sandbox.** This bounds tools by what they do; it doesn't stop a tool doing
something other than what we filed it under. `execute` can write files. That's an argument
for capping `execute`, not for distrusting the grouping.
"""
from __future__ import annotations

READ = "read"
WRITE = "write"
EXECUTE = "execute"
SPAWN = "spawn"

TOOL_CAPABILITIES: dict[str, str] = {
    # Exactly deepagents' `_DEFAULT_FS_TOOL_OPS`, names and groupings both.
    "ls": READ,
    "read_file": READ,
    "glob": READ,
    "grep": READ,
    "write_file": WRITE,
    "edit_file": WRITE,
    "delete": WRITE,
    # Ours: deepagents ships these but classifies neither.
    "execute": EXECUTE,
    "task": SPAWN,
}


def register_capability(tool: str, capability: str) -> None:
    """File a tool under a capability — a customer's own tool, or one from a harness
    release newer than this map.

    Process-wide and additive, so it belongs at import time next to the tool's
    definition rather than inside a workflow handler.
    """
    TOOL_CAPABILITIES[tool] = capability


def capability_of(tool: str) -> str | None:
    """The capability a tool is filed under, or None if it isn't classified.

    Unclassified is not a failure. A capability cap simply doesn't reach the tool, which
    is why an allowlist is the mechanism for "nothing but these" — capabilities bound
    what you know about, allowlists bound what you don't.
    """
    return TOOL_CAPABILITIES.get(tool)


def tools_with(capability: str) -> set[str]:
    """Every tool currently filed under a capability."""
    return {t for t, c in TOOL_CAPABILITIES.items() if c == capability}


def file_permissions(policy) -> list:
    """Turn a `RuntimePolicy`'s `file_rules` into deepagents' `FilesystemPermission`s.

        agent = create_deep_agent(..., permissions=file_permissions(governor.policy))

    The same division as everywhere else: BoundFlow says which files matter and what
    should happen, the harness does the matching. That matching is worth not owning — it
    handles bulk tools (`ls`/`glob`/`grep`) by whether their search subtree could overlap
    a rule, and analyses recursive deletes against every deny pattern that could cover a
    descendant. Reimplementing it would mean reimplementing those subtleties, wrongly.

    A rule with `mode="interrupt"` becomes a durable gate rather than an in-process one,
    since the interrupt it raises is the same interrupt `harness_gates.pending_action`
    reads.
    """
    from deepagents.middleware.filesystem import FilesystemPermission

    return [FilesystemPermission(operations=list(rule.operations),
                                 paths=list(rule.paths), mode=rule.mode)
            for rule in policy.file_rules]
