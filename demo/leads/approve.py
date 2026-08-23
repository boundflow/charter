"""Sign off on whatever the outreach agents are waiting for.

Charter has no `approve` command — approving is a control-plane action, and the
CLI deliberately doesn't reach past it. So this is a small operator's console
built on the same public API a real one would use.

    python demo/leads/approve.py            # watch, and prompt for each gate
    python demo/leads/approve.py --auto     # approve everything, for a hands-off run

Each gate shows the justification the agent gave, which is the tool call it wants
to make and why. Approving lets that exact call through; rejecting ends the task,
because outreach declares `on_reject: fail` — a run that reported success without
sending the message would be lying.
"""

import argparse
import asyncio
import os
import sys

from boundflow import ControlPlaneClient

POLL_SECONDS = 3


async def pending(cp, tenant_id: str):
    """Every workflow currently stopped for a human, with its open gate.

    `list_workflows` returns a light view without the gate itself, so anything
    that looks stopped is fetched in full to find out what it is asking for.
    """
    out = []
    for w in await cp.list_workflows():
        if w.tenant_id != tenant_id:
            continue
        state = getattr(w.lifecycle_state, "name", str(w.lifecycle_state))
        if "AWAITING" not in state.upper():
            continue
        full = await cp.get_workflow(w.id)
        if full.pending_approval is not None:
            out.append((full, full.pending_approval))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true",
                    help="approve everything without asking")
    ap.add_argument("--tenant", default="default")
    args = ap.parse_args()

    cp = ControlPlaneClient(os.environ["BOUNDFLOW_SERVER_ADDRESS"],
                            os.environ["BOUNDFLOW_API_KEY"])
    async with cp:
        try:
            tenants = await cp.list_tenants()
        except Exception as e:  # noqa: BLE001
            # Fail here rather than retry: if the control plane is unreachable at
            # startup that is a wrong address or a server that isn't up, and neither
            # gets better by waiting.
            sys.exit(f"can't reach the control plane at "
                     f"{os.environ['BOUNDFLOW_SERVER_ADDRESS']}: "
                     f"{e.__class__.__name__}")
        tenant = next((t for t in tenants if t.name == args.tenant), None)
        if tenant is None:
            sys.exit(f"no tenant named {args.tenant!r}")

        print(f"watching {args.tenant} for approvals "
              f"({'auto-approving' if args.auto else 'interactive'}) — ctrl-c to stop\n")
        seen: set[str] = set()
        while True:
            try:
                waiting = await pending(cp, tenant.id)
            except Exception as e:  # noqa: BLE001
                # Keep watching. A console that exits on a blip leaves gates sitting
                # unapproved with nobody looking and nothing saying it stopped —
                # which reads exactly like an agent that has gone quiet.
                print(f"   control plane unreachable ({e.__class__.__name__}), "
                      f"retrying")
                await asyncio.sleep(POLL_SECONDS)
                continue

            for workflow, gate in waiting:
                if gate.approval_id in seen:
                    continue
                seen.add(gate.approval_id)
                print(f"── {workflow.workflow_type} {workflow.id[:8]}")
                print(f"   {gate.justification}\n")
                if args.auto:
                    verdict = "y"
                else:
                    verdict = input("   approve? [y/N] ").strip().lower()
                if verdict == "y":
                    await cp.approve_workflow(workflow.id, gate.approval_id,
                                              actor="demo", reason="looks good")
                    print("   approved\n")
                else:
                    await cp.reject_workflow(workflow.id, gate.approval_id,
                                             actor="demo", reason="not sending that")
                    print("   rejected\n")
            await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped watching")
