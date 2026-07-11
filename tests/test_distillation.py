"""Distillation placeholder tests (Milestone 0).

Distillation is intentionally NOT implemented in Milestone 0. These tests assert the module
exists as a guarded placeholder so accidental use fails loudly, and mark the real behavior as
expected-to-come with an xfail.
"""

from __future__ import annotations

import pytest

from coordinator_bert import distillation


def test_distiller_is_placeholder_and_blocks_use():
    """Instantiating the placeholder must raise NotImplementedError, not silently work."""
    with pytest.raises(NotImplementedError):
        distillation.TeacherDistiller()


@pytest.mark.xfail(reason="Teacher distillation is a Milestone 2 feature; not implemented yet.",
                   strict=True)
def test_distillation_implemented_in_future():
    distiller = distillation.TeacherDistiller()  # will raise until Milestone 2
    assert distiller is not None
