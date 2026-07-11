"""Teacher distillation — PLACEHOLDER (not implemented in Milestone 0).

Milestone 2 will use a larger teacher (e.g. Qwen) to teach coordination-relevant
representations to the small student encoder. This module deliberately contains no logic yet;
it exists so the package layout is stable and future work has a clear home.

Planned (future) responsibilities:
  * load / query a frozen teacher model,
  * representation-alignment loss (hidden-state / attention matching),
  * soft-label (KL) distillation on top of MLM,
  * loss weighting / scheduling between MLM and distillation terms.

Keeping distillation strictly separate from base MLM pretraining is a hard project rule.
"""

from __future__ import annotations


class TeacherDistiller:
    """Placeholder. Instantiation is blocked until Milestone 2."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        raise NotImplementedError(
            "Teacher distillation is not implemented in Milestone 0. "
            "See dev_mem/project_brief.md (Milestone 2)."
        )
