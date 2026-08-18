"""Terminal output.

Plain, aligned, greppable — closer to kubectl than to a dashboard. Someone reading
this is usually mid-incident or mid-deploy, so the priorities are: scannable in one
glance, pipeable into grep and awk, and quiet enough that colour means something
when it does appear.

Colour is load-bearing rather than decorative: red is an error, yellow is something
that needs a person, dim is context. Everything else is plain.
"""

from __future__ import annotations

import re
import sys

import typer

# Colour is invisible but not zero-width to str.ljust, which pads by byte count
# and silently ruins every column after a styled cell.
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _visible(text: str) -> int:
    return len(ANSI.sub("", text))


def _pad(text: str, width: int) -> str:
    return text + " " * max(width - _visible(text), 0)

# kubectl's convention, and it reads well: `agent/refund-demo approved`.
def ref(kind: str, name: str) -> str:
    return typer.style(f"{kind}/{name}", bold=True)


def err(msg: str) -> None:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)


def detail(msg: str) -> None:
    """An indented continuation of the line above — usually what to do about it."""
    typer.secho(f"  {msg}", fg=typer.colors.BRIGHT_BLACK, err=not sys.stdout.isatty())


def warn(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.YELLOW)


def ok(msg: str) -> None:
    typer.echo(msg)


def dim(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.BRIGHT_BLACK)


def table(headers: list[str], rows: list[list[str]], notes: list[str] | None = None) -> None:
    """Left-aligned columns sized to content, two spaces between. No borders — they
    survive neither a narrow terminal nor a pipe into awk.

    `notes` prints one dim line under its row — for anything too long to be a
    column, like a failure reason, which would otherwise compete with a 36-char
    identifier for width and always lose.
    """
    if not rows:
        return
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(_visible(r[i]) for r in cells)) for i, h in enumerate(headers)]

    typer.secho("  ".join(h.upper().ljust(w) for h, w in zip(headers, widths)).rstrip(),
                fg=typer.colors.BRIGHT_BLACK)
    for i, row in enumerate(cells):
        typer.echo("  ".join(_pad(c, w) for c, w in zip(row, widths)).rstrip())
        # A long message belongs under its row, not in a column: as a column it
        # competes with the identifier for width and always loses.
        if notes and i < len(notes) and notes[i]:
            typer.secho(f"    {notes[i]}", fg=typer.colors.BRIGHT_BLACK)


def kv(pairs: list[tuple[str, object]], indent: str = "") -> None:
    """A block of label/value lines, labels aligned."""
    if not pairs:
        return
    width = max(len(k) for k, _ in pairs)
    for key, value in pairs:
        typer.echo(f"{indent}{key.ljust(width)}   {value}")


# Two orthogonal state machines, and conflating them hides the important one.
#
#   workflow_state   may it work at all — active | paused | cooldown | disabled.
#                    This is what the scheduler checks; a lifecycle rule moves it.
#   lifecycle_state  what it's doing right now — invoking, awaiting_approval, ...
#
# A paused agent can sit at lifecycle_state "active", so showing only that would
# report a stopped agent as healthy.
STOPPED = {"paused", "disabled"}
NEEDS_ATTENTION = {"awaiting_approval", "awaiting_input", "blocked", "failed",
                   "interrupted", "cooldown"}


def state(value: str) -> str:
    if value in STOPPED:
        return typer.style(value, fg=typer.colors.RED)
    if value in NEEDS_ATTENTION:
        return typer.style(value, fg=typer.colors.YELLOW)
    return value


def working(workflow_state: str) -> bool:
    """Only `active` schedules; everything else means no new tasks start."""
    return workflow_state == "active"


def activity(lifecycle_state: str) -> str:
    """lifecycle_state 'active' means "idle, no run in flight" — BoundFlow's own
    docs define it that way. Printed verbatim next to a workflow_state of 'paused'
    it reads as a contradiction, so it's shown as what it means."""
    return "idle" if lifecycle_state == "active" else lifecycle_state


def gate(agent: str, kind: str, gate_id: str, body: str, actions: list[str],
         timeout: str = "") -> None:
    """The one screen that should slow you down.

    Everything else here is built to be skimmed; this is a person deciding whether
    a machine may do something irreversible. So it gets a rule, a subject set apart
    from the chrome, and the exact commands — no borders or colour beyond the one
    marker, because it still has to survive a pipe and a narrow terminal.
    """
    width = min(_term_width(), 78)
    typer.echo()
    typer.secho("─" * width, fg=typer.colors.YELLOW)
    head = f"{agent} needs {kind}"
    typer.secho(f" {head}", fg=typer.colors.YELLOW, bold=True)
    if timeout:
        typer.secho(f" expires {timeout}", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("─" * width, fg=typer.colors.YELLOW)
    typer.echo()
    for line in body.strip().splitlines():
        typer.echo(f"   {line}")
    typer.echo()
    for action in actions:
        typer.secho(f"   {action}", fg=typer.colors.BRIGHT_BLACK)
    typer.echo()


def _term_width() -> int:
    import shutil
    return shutil.get_terminal_size((80, 24)).columns


def worker_banner(name: str, tenant: str, rows: list[tuple[str, int, int, int, str]]) -> None:
    """What this process is serving, printed once at boot.

    A worker is a pane someone leaves open, so the question it should answer at a
    glance is "is anything broken" — which is why a quarantined agent shows its
    reason here rather than only in a log line that has already scrolled past.
    """
    typer.echo()
    typer.secho(f"charter worker {name}", bold=True)
    typer.secho(f"tenant {tenant}", fg=typer.colors.BRIGHT_BLACK)
    typer.echo()
    table(["agent", "ver", "tools", "gated", "status"],
          [[a, f"v{v}", str(t), str(g),
            "ready" if s == "ready" else "quarantined"] for a, v, t, g, s in rows])
    broken = [(a, s) for a, _, _, _, s in rows if s != "ready"]
    for agent, reason in broken:
        typer.echo()
        warn(f"{agent} is quarantined — its tasks will fail fast")
        detail(reason)
    typer.echo()
