"""What `charter init` writes.

The templates live here rather than in the CLI so the documentation test can hold
the README's quickstart against them. A scaffold that drifts from the README it is
documented by is worse than no scaffold: the reader follows the prose, the command
writes something else, and the mismatch surfaces as a validation error against a
file they never typed.

Two files, matching the quickstart exactly: an agent that calls no tools and sets
no budget, and the worker manifest that serves it.
"""

from __future__ import annotations

AGENT = """\
apiVersion: charter/v1
kind: AgentConfig

name: {name}
version: 1
model: claude-haiku-4-5

objective: |
  Triage this support ticket and say what should happen to it:

  {{{{ inputs.ticket }}}}

inputs:
  ticket: {{ type: string, required: true }}

response_format:
  category:
    type: string
    description: billing, bug, account, or other.
  next_step:
    type: string
    description: What a person should do about it, in one sentence.
"""

WORKER = """\
apiVersion: charter/v1
kind: Worker

control_plane:
  endpoint: ${{BOUNDFLOW_SERVER_ADDRESS}}
  worker_endpoint: ${{BOUNDFLOW_WORKER_ADDRESS}}
  api_key: ${{BOUNDFLOW_API_KEY}}
  tenant: default

llm:
  # Any provider LangChain can build. Name it here, put its key in the variable
  # below, and install its package: pip install 'boundflow-charter[openai]'
  provider: anthropic
  api_key: ${{ANTHROPIC_API_KEY}}

store:
  url: ${{CHARTER_STORE_URL}}

agents_dir: ./
serves:
  - agent: {name}
    versions: [1]
"""


def files(name: str) -> dict[str, str]:
    """Path relative to the project directory, mapped to its contents."""
    return {f"{name}/v1.yaml": AGENT.format(name=name),
            "worker.yaml": WORKER.format(name=name)}


SERVES_ENTRY = "  - agent: {name}\n    versions: [1]\n"


def add_to_serves(text: str, name: str) -> str:
    """Append an agent to an existing manifest's `serves:` list.

    Edits the text rather than reserialising the parsed document, so comments,
    key order and formatting survive. A manifest people are told to edit by hand
    is one a tool has no business rewriting.
    """
    lines = text.splitlines(keepends=True)
    try:
        start = next(i for i, l in enumerate(lines) if l.rstrip() == "serves:")
    except StopIteration:
        raise ValueError("no `serves:` list to add to") from None

    # The block runs to the next line that starts its own top-level key. Blank
    # lines and comments inside it belong to it.
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped and not lines[i][:1].isspace():
            end = i
            break

    # Trailing blanks belong after the insertion, not before it.
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1

    entry = SERVES_ENTRY.format(name=name)
    if not lines[end - 1].endswith("\n"):
        lines[end - 1] += "\n"
    return "".join(lines[:end]) + entry + "".join(lines[end:])
