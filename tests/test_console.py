"""Charter's wording on BoundFlow's console.

The console renders whatever `Labels` says, so these pin the two halves an operator
moves between: the words match `charter agents`, and no BoundFlow noun survives
into a Charter screen.
"""
import re
from dataclasses import fields

from charter.console import LABELS

# BoundFlow's nouns for the things Charter renames. Substring matching would trip
# on "runtime", so these are whole words.
THEIRS = re.compile(r"\b(workflow|workflows|run|runs|lifecycle)\b", re.I)


def test_the_console_says_what_charter_agents_says():
    """`charter agents` heads its columns agent, instance, status and activity, and
    calls one invocation a task. The agent is the workflow *type*; the workflow
    itself is the instance — swapping those two is the mistake worth pinning."""
    assert (LABELS.workflow, LABELS.workflow_type) == ("instance", "agent")
    assert (LABELS.state, LABELS.lifecycle) == ("status", "activity")
    assert (LABELS.run, LABELS.runs) == ("task", "tasks")


def test_no_boundflow_noun_reaches_a_charter_screen():
    """Every Labels field defaults to BoundFlow's own wording, so one added upstream
    arrives untranslated. This fails then, rather than shipping a console that says
    workflow on one page and agent on the next."""
    leaked = {f.name: getattr(LABELS, f.name) for f in fields(LABELS)
              if THEIRS.search(str(getattr(LABELS, f.name)))}
    assert not leaked, f"untranslated: {leaked}"


def test_the_brand_is_ours():
    assert LABELS.brand == "Charter"
