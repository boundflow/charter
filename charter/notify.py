"""Outbound webhook delivery for gates.

An approval nobody hears about is just a slow timeout, so this is load-bearing
rather than a nicety.

One primitive: a signed webhook. Not Slack/PagerDuty/email integrations — all of
them ingest webhooks, as do Zapier and n8n, and a URL plus a secret is the only
credential material Charter has to hold. Staying out of the integrations business
is worth more than the convenience.

Delivery is best-effort. A failed notification never fails the task: the gate still
exists and degrades to its declared on_timeout branch, which the state machine
already handles. Failures log loudly, because "the gate opened and nobody was told"
is exactly what an operator needs to find out about.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import urllib.error
import urllib.request

from .config.worker import Channel, Notifications

log = logging.getLogger(__name__)

ENV_REF = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def _resolve(value: str) -> str:
    return ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value or "")


def sign(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class Notifier:
    """Routes gate events to channels. Routing is keyed by agent NAME — a string,
    not a path — so a worker manifest never reaches into an agent's repo."""

    def __init__(self, notifications: Notifications | None) -> None:
        self._notifications = notifications

    async def approval_requested(self, request) -> None:
        # Logged before routing, always. A worker with no channels configured would
        # otherwise swallow the one identifier needed to unblock the task.
        log.info("APPROVAL %s needs a decision: %s",
                 request.approval_id, (request.justification or "").splitlines()[:1])
        await self._send("approval_requested", request, request.approval_id, {
            "approval_id": request.approval_id,
            "justification": request.justification or "",
            "timeout_seconds": request.timeout,
        })

    async def input_requested(self, request) -> None:
        log.info("INPUT %s needs an answer: %s", request.input_id, request.prompt)
        await self._send("input_requested", request, request.input_id, {
            "input_id": request.input_id,
            "question": request.prompt or "",
            "timeout_seconds": request.timeout,
        })

    async def _send(self, event: str, request, gate_id: str, extra: dict) -> None:
        agent = (request.metadata or {}).get("agent") or ""

        # Both undelivered paths carry the gate id. This is precisely when someone
        # needs it — the task is parked and nothing is going to arrive to tell them.
        if self._notifications is None:
            log.warning("%s %s for %s undelivered: this worker has no notification "
                        "channels. Unblock it with: charter pending %s",
                        event, gate_id, agent or request.workflow_id, agent)
            return

        channel = self._notifications.resolve(agent, event)
        if channel is None:
            log.warning("%s %s for %s undelivered: no route matches. Unblock it "
                        "with: charter pending %s", event, gate_id, agent, agent)
            return

        payload = {
            "event": event,
            "agent": agent,
            "workflow_id": request.workflow_id,
            "operation": request.operation_name,
            **extra,
        }
        await self._post(channel, payload)

    async def _post(self, channel: Channel, payload: dict) -> None:
        body = json.dumps(_shape(channel, payload)).encode()
        headers = {"Content-Type": "application/json"}
        if channel.secret:
            headers["X-Charter-Signature"] = sign(_resolve(channel.secret), body)

        url = _resolve(channel.url)
        for attempt in range(1, channel.max_attempts + 1):
            try:
                await asyncio.to_thread(_post_once, url, body, headers,
                                        channel.timeout_seconds)
                return
            except Exception as e:  # noqa: BLE001
                if attempt == channel.max_attempts:
                    log.error("notification to %s failed after %d attempts (%s) — "
                              "gate %s is open and nobody was told",
                              channel.name, attempt, e, payload.get("approval_id")
                              or payload.get("input_id"))
                    return
                await asyncio.sleep(2 ** (attempt - 1))


def _shape(channel: Channel, payload: dict) -> dict:
    """The same message in whatever body the destination accepts."""
    if channel.kind == "webhook":
        return payload
    text = _text(payload)
    if channel.kind == "telegram":
        return {"chat_id": _resolve(channel.chat_id or ""), "text": text,
                "parse_mode": "Markdown"}
    return {"text": text}


def _text(payload: dict) -> str:
    """One message: what needs deciding, then the command that decides it. Someone
    reading this on a phone shouldn't have to go and look up an id."""
    agent = payload.get("agent") or "an agent"
    gate = payload.get("approval_id") or payload.get("input_id") or ""
    if payload["event"] == "approval_requested":
        lines = [f"*{agent}* needs approval", "```", payload.get("justification", ""), "```",
                 f"`charter approve {gate} --agent {agent} --reason \"...\"`",
                 f"`charter reject  {gate} --agent {agent} --reason \"...\"`"]
    else:
        lines = [f"*{agent}* is asking:", "```", payload.get("question", ""), "```",
                 f"`charter answer {gate} \"...\" --agent {agent}`"]
    return "\n".join(lines)


def _post_once(url: str, body: bytes, headers: dict, timeout: int) -> None:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status >= 400:
            raise urllib.error.HTTPError(url, resp.status, "bad status", None, None)
