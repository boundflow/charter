"""Charter's words on BoundFlow's operator console.

`charter ui` serves the console from `boundflow.ui` against the control plane this
project already points at. Only the console's own wording changes: an agent, a
task, and the status and activity columns `charter agents` prints.

What stays BoundFlow's is anything the control plane returns — `awaiting_approval`,
`cooldown`, a workflow type. Those are what `charter describe` prints and what lands
in the audit log, so an operator who reads one here can search for it there.
"""

from __future__ import annotations

from boundflow.ui import Labels

LABELS = Labels(
    brand="Charter",
    tagline="local agent console",
    # A BoundFlow workflow is one Charter instance, and its type is the agent —
    # the columns `charter agents` heads instance and agent.
    workflow="instance",
    workflows="instances",
    workflow_type="agent",
    # `charter agents` heads these columns status and activity.
    lifecycle="activity",
    state="status",
    run="task",
    runs="tasks",
    fleet="Agents",
    abandon="Abandon queued tasks",
    empty_fleet="No agents yet.",
    empty_runs="No tasks yet.",
)
