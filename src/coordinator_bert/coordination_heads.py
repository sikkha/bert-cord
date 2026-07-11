"""Coordination heads — PLACEHOLDER (not implemented in Milestone 0).

Milestone 3 will add factorized decision heads on top of the pooled encoder representation so
the coordinator can decide *what to do* with an event without performing the full reasoning
task itself.

Planned (future) factorized heads:
  * target organ (which specialist / memory / core LLM),
  * operation,
  * priority,
  * control transition (continue / pause / interrupt / stop / resume / terminate),
  * context-transfer policy (what state to carry between AI units).

No logic exists yet; this module is a stable home for that future work.
"""

from __future__ import annotations


class CoordinationHeads:
    """Placeholder. Instantiation is blocked until Milestone 3."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        raise NotImplementedError(
            "Coordination heads are not implemented in Milestone 0. "
            "See dev_mem/project_brief.md (Milestone 3)."
        )
