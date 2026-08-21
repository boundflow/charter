"""Take the harness's own numbers as the truth about what a run spent.

BoundFlow counts model calls as it makes them, which has two problems. The count lives
in worker memory until the operation ends, so a crash takes the accounting with it. And
it only covers calls that went through us — a subagent whose spec names its model as a
string builds its own client, and that spend is invisible.

The harness's state has neither problem. Every `AIMessage` it writes carries
`usage_metadata` — exact input and output tokens, the cache split, and the model that
produced them — and it writes them whoever made the call. Metering the checkpointer puts
us on that write:

    saver = metered(AsyncPostgresSaver(...), governor)

so metering is a property of the wiring rather than something a caller remembers to do.

The rule this encodes: **where both sides can produce a number, the harness's wins.**
Tokens, calls and models come from here. What stays ours is what they structurally can't
know — the price of a token, whether a call was permitted, and which agent and version
any of it belongs to.

Not covered, and no amount of wrapping fixes it: a call the provider billed whose
response never reached the checkpoint, because the process died in between. Bounded to
one call, and the only cure is reserving before spending — see `GovernedCall`.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def metered(saver: Any, governor: Any, report: Any = None) -> Any:
    """Record usage from everything `saver` writes. Returns the same saver.

    It installs a subclass onto the instance rather than wrapping it. A wrapper would
    have to be a `BaseCheckpointSaver` — LangGraph type-checks it — and would then
    inherit that base's default implementations for every method it forgot to forward,
    silently replacing whatever the real saver overrode. Subclassing the saver's own
    class can't have that bug: anything not mentioned here is untouched.
    """
    if getattr(saver, "_boundflow_metered", False):
        return saver

    governor.register_harness_metering()
    # A message appears in the writes and again in every later checkpoint of the same
    # thread. Counting it each time would multiply a run's spend by the number of
    # super-steps, so each id is only ever counted once.
    counted: set[str] = set()

    def meter(values: Any) -> None:
        """Record usage for any model message not already seen.

        Never raises. A metering bug must not be able to lose a customer's state — the
        write it rides on is the only durable copy of the agent's progress.
        """
        try:
            for message in _messages(values):
                usage = getattr(message, "usage_metadata", None)
                key = getattr(message, "id", None)
                if not usage or key is None or key in counted:
                    continue
                counted.add(key)
                governor.record_harness_usage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    details=usage.get("input_token_details") or {},
                    model=(getattr(message, "response_metadata", None) or {}).get("model_name"),
                )
        except Exception:  # noqa: BLE001
            log.warning("failed to meter a checkpoint write", exc_info=True)

    async def _report() -> None:
        # Actuals replace the reservation server-side as soon as they're known.
        if report is not None:
            await report()

    def _checkpoint_messages(checkpoint) -> Any:
        return (checkpoint or {}).get("channel_values", {}).get("messages")

    base = type(saver)

    class MeteredSaver(base):  # type: ignore[misc, valid-type]
        """The saver it already was, plus metering on the way through."""

        _boundflow_metered = True

        async def aput(self, config, checkpoint, metadata, new_versions):
            meter(_checkpoint_messages(checkpoint))
            await _report()
            return await super().aput(config, checkpoint, metadata, new_versions)

        async def aput_writes(self, config, writes, task_id, task_path=""):
            # Where a parent agent's calls are seen, and the only place: its `aput`
            # checkpoints are deltas whose channel_values are empty. (A subagent writes
            # its whole state, so it shows up in both.)
            meter([value for _channel, value in writes])
            await _report()
            return await super().aput_writes(config, writes, task_id, task_path)

        def put(self, config, checkpoint, metadata, new_versions):
            meter(_checkpoint_messages(checkpoint))
            return super().put(config, checkpoint, metadata, new_versions)

        def put_writes(self, config, writes, task_id, task_path=""):
            meter([value for _channel, value in writes])
            return super().put_writes(config, writes, task_id, task_path)

    MeteredSaver.__name__ = f"Metered{base.__name__}"
    saver.__class__ = MeteredSaver
    return saver


def _messages(values: Any):
    """Every message reachable in a checkpoint value, whatever shape it arrived in.

    Recurses, because the shapes nest and the nesting is where this goes wrong: a write
    is a `(channel, value)` pair whose value is usually a *list* of messages, so the
    messages sit two levels down. Handle only one level and a parent agent's calls are
    silently skipped while its subagents' are counted — the sort of half-working that
    reads as a plausible number rather than as a bug.

    Channels other than `messages` carry things that aren't messages at all, hence
    matching on the message's own attributes rather than on the channel name.
    """
    if values is None:
        return
    if isinstance(values, dict):
        yield from _messages(values.get("messages"))
        return
    if hasattr(values, "usage_metadata") or hasattr(values, "tool_calls"):
        yield values
        return
    if isinstance(values, (list, tuple)):
        for value in values:
            yield from _messages(value)
