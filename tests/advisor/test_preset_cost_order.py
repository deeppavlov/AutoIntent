"""``PRESET_COST_ORDER`` must stay in sync with the preset literal.

``recommend`` picks the heaviest preset that fits, so a preset missing from the
ranking would silently sort last and effectively never be recommended. This test
turns that into a hard failure at the moment a preset is added.
"""

from __future__ import annotations

from typing import get_args

from autointent.advisor._workflows import PRESET_COST_ORDER
from autointent.custom_types import SearchSpacePreset


def test_cost_order_covers_every_preset() -> None:
    assert set(PRESET_COST_ORDER) == set(get_args(SearchSpacePreset))


def test_cost_order_has_no_duplicates() -> None:
    assert len(PRESET_COST_ORDER) == len(set(PRESET_COST_ORDER))


def test_preset_literal_order_is_not_load_bearing() -> None:
    """The literal's declaration order must not be the cost ranking.

    Cost ordering belongs in PRESET_COST_ORDER, where it is commented and
    tested. If these two ever coincide exactly, someone has reintroduced the
    coupling -- the literal is kept in dev's original (roughly alphabetical)
    order precisely so it cannot be mistaken for a ranking.
    """
    assert tuple(get_args(SearchSpacePreset)) != PRESET_COST_ORDER
