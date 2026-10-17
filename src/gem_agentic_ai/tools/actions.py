"""SIDE-EFFECTING DEMO TOOL
========================
One consequential action, used to demonstrate approval state and idempotency.

WHY a ticket rather than a real integration?
  The action must be observable but harmless: it appends to an in-process
  ledger so tests can prove that exactly one ticket exists. Nothing leaves
  the machine and no credentials are involved.

WHO ENFORCES IDEMPOTENCY
  The tool does not de-duplicate its own calls. ``ToolGuard`` refuses a
  second execution for the same action key and returns the recorded result.
  That keeps the policy in one place instead of scattering it through every
  side-effecting tool. Calling this tool directly, without a guard, bypasses
  the protection — which is why the agent builders wrap every tool.
"""

import logging
import threading

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_LEDGER: list[dict[str, str]] = []
_LOCK = threading.Lock()


@tool
def create_support_ticket(summary: str, idempotency_key: str) -> str:
    """Create a support ticket for the user's request.

    This is a consequential action: it records a ticket that a support team
    would act on. The caller must supply an ``idempotency_key`` so a retried
    or replayed agent run produces one logical ticket rather than several.

    Args:
        summary: One-line description of the user's issue.
        idempotency_key: Stable identifier for this logical action. Reusing
            the key returns the earlier result instead of creating a second
            ticket.

    Returns:
        A confirmation naming the ticket's idempotency key.
    """
    ticket = {"idempotency_key": idempotency_key, "summary": summary}
    with _LOCK:
        _LEDGER.append(ticket)
    logger.info(
        "Created demo support ticket for key '%s' (ledger size %d)",
        idempotency_key,
        len(_LEDGER),
    )
    return f"Ticket created for '{idempotency_key}': {summary}"


def ticket_ledger() -> tuple[dict[str, str], ...]:
    """Return a snapshot of the demo ledger (oldest first)."""
    with _LOCK:
        return tuple(dict(item) for item in _LEDGER)


def reset_ticket_ledger() -> None:
    """Clear the demo ledger. Intended for tests."""
    with _LOCK:
        _LEDGER.clear()
