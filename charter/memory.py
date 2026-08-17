"""Memory derived from the governance audit log.

No store, no embeddings, no extraction pass. Every approval decision and every
answered question is already recorded durably, so the corpus exists — and unlike
summarized conversation history, every row is a *human judgment about this specific
agent*. "Here are the last ten times a person told this agent it was wrong, in
their words."

Two properties that fall out of deriving rather than storing:

  * It is exactly inspectable. `charter memory <agent>` prints the same text the
    agent will be shown, which no embedding index can offer.
  * It cannot drift from what actually happened, because it *is* what happened.

This covers memory of human judgments about the agent. Memory of the *world* —
documents, customer history, semantic recall — is an MCP server with a read tool
and a write tool, governed like any other. Charter ships no retrieval.
"""

from __future__ import annotations

import logging

from boundflow import ApprovalDecision, ControlPlaneClient, InputDecision

from .config.agent import AgentConfig

log = logging.getLogger(__name__)


class AuditMemory:
    """Recalls a workflow's human judgments, newest first.

    Cached per task: `recall` runs on every round, and the audit log only changes
    when a human decides something — which can't happen mid-round, since the task
    is what would be parked waiting for them.
    """

    def __init__(self, cp: ControlPlaneClient) -> None:
        self._cp = cp
        self._cache: dict[str, list[str]] = {}

    async def recall(self, cfg: AgentConfig, workflow_id: str = "") -> list[str]:
        spec = cfg.memory.from_audit if cfg.memory else None
        if spec is None or not (spec.rejections or spec.answers) or not workflow_id:
            return []
        if workflow_id not in self._cache:
            self._cache[workflow_id] = await self._load(
                workflow_id, spec.rejections, spec.answers)
        return self._cache[workflow_id]

    async def _load(self, workflow_id: str, want_rejections: int,
                    want_answers: int) -> list[str]:
        try:
            entries = await self._cp.get_audit_log(workflow_id=workflow_id)
        except Exception:  # noqa: BLE001
            # Memory is an improvement, not a dependency. An agent that can't read
            # its history should still do the task.
            log.warning("could not load audit memory for %s", workflow_id, exc_info=True)
            return []

        rejections: list[str] = []
        answers: list[str] = []

        for entry in entries:  # newest first
            decision = getattr(entry, "decision", None)

            if decision == ApprovalDecision.REJECTED and len(rejections) < want_rejections:
                rejections.append(_rejection(entry))
            elif decision == InputDecision.ANSWERED and len(answers) < want_answers:
                if line := _answer(entry):
                    answers.append(line)

            if len(rejections) >= want_rejections and len(answers) >= want_answers:
                break

        # Rejections first: being told you were wrong is stronger guidance than
        # being told a fact, and the agent reads top-down.
        return rejections + answers


def _rejection(record) -> str:
    """A rejection is only useful with its subject attached — "rejected" alone says
    nothing about what not to do again."""
    what = record.justification or "a previous proposal"
    line = f'REJECTED: {_oneline(what)}'
    if record.reason:
        line += f' — because: "{_oneline(record.reason)}"'
    return line


def _answer(record) -> str:
    question = _oneline(record.prompt) or "a question"
    text = (record.answer or {}).get("text", "")
    if not text:
        return ""
    return f'ASKED: "{question}" — ANSWERED: "{_oneline(text)}"'


def _oneline(text: str, limit: int = 300) -> str:
    """Memory is prepended to every round's prompt, so a long justification would
    be paid for on every LLM call of every task."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
