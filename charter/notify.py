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
        await self._send("approval_requested", request, {
            "approval_id": request.approval_id,
            "justification": request.justification or "",
            "timeout_seconds": request.timeout,
        })

    async def input_requested(self, request) -> None:
        await self._send("input_requested", request, {
            "input_id": request.input_id,
            "question": request.prompt or "",
            "timeout_seconds": request.timeout,
        })

    async def _send(self, event: str, request, extra: dict) -> None:
        if self._notifications is None:
            log.warning("%s for workflow %s but this worker has no notification "
                        "channels — nobody was told", event, request.workflow_id)
            return

        agent = (request.metadata or {}).get("agent") or ""
        channel = self._notifications.resolve(agent, event)
        if channel is None:
            log.warning("%s for %s but no route matches — nobody was told", event, agent)
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
        body = json.dumps(payload).encode()
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
                              "the gate is open and nobody was told",
                              channel.name, attempt, e)
                    return
                await asyncio.sleep(2 ** (attempt - 1))


def _post_once(url: str, body: bytes, headers: dict, timeout: int) -> None:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status >= 400:
            raise urllib.error.HTTPError(url, resp.status, "bad status", None, None)
