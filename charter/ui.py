"""Terminal output.

Plain, aligned, greppable — closer to kubectl than to a dashboard. Someone reading
this is usually mid-incident or mid-deploy, so the priorities are: scannable in one
glance, pipeable into grep and awk, and quiet enough that colour means something
when it does appear.

Colour is load-bearing rather than decorative: red is an error, yellow is something
that needs a person, dim is context. Everything else is plain.
"""

from __future__ import annotations

import sys

import typer

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


def table(headers: list[str], rows: list[list[str]]) -> None:
    """Left-aligned columns sized to content, two spaces between. No borders — they
    survive neither a narrow terminal nor a pipe into awk."""
    if not rows:
        return
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    typer.secho("  ".join(h.upper().ljust(w) for h, w in zip(headers, widths)).rstrip(),
                fg=typer.colors.BRIGHT_BLACK)
    for row in cells:
        typer.echo("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())


def kv(pairs: list[tuple[str, object]], indent: str = "") -> None:
    """A block of label/value lines, labels aligned."""
    if not pairs:
        return
    width = max(len(k) for k, _ in pairs)
    for key, value in pairs:
        typer.echo(f"{indent}{key.ljust(width)}   {value}")


# States that mean a person has to do something, so they're worth colouring.
NEEDS_ATTENTION = {"awaiting_approval", "awaiting_input", "blocked", "failed"}


def state(value: str) -> str:
    if value in NEEDS_ATTENTION:
        return typer.style(value, fg=typer.colors.YELLOW)
    return value


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
