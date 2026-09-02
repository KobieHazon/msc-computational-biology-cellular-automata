"""Neighborhood index groups used by the stripe-forming rules."""

from typing import ClassVar

CellIndex = tuple[int, int]


class CellNeighborhoodIndices:
    """Immutable index sets for a 3-by-3 wrapped neighborhood."""

    RIGHT_INDICES: ClassVar[frozenset[CellIndex]] = frozenset({(0, 2), (1, 2), (2, 2)})
    LEFT_INDICES: ClassVar[frozenset[CellIndex]] = frozenset({(0, 0), (1, 0), (2, 0)})
    UPPER_INDICES: ClassVar[frozenset[CellIndex]] = frozenset({(0, 0), (0, 1), (0, 2)})
    LOWER_INDICES: ClassVar[frozenset[CellIndex]] = frozenset({(2, 0), (2, 1), (2, 2)})

    UPPER_RIGHT_INDICES = UPPER_INDICES | RIGHT_INDICES
    LOWER_RIGHT_INDICES = LOWER_INDICES | RIGHT_INDICES
    UPPER_LEFT_INDICES = UPPER_INDICES | LEFT_INDICES
    LOWER_LEFT_INDICES = LOWER_INDICES | LEFT_INDICES

    UPPER_RIGHT_CORNER_INDICES = UPPER_RIGHT_INDICES.difference(LEFT_INDICES | LOWER_INDICES)
    LOWER_RIGHT_CORNER_INDICES = LOWER_RIGHT_INDICES.difference(UPPER_INDICES | LEFT_INDICES)
    UPPER_LEFT_CORNER_INDICES = UPPER_LEFT_INDICES.difference(LOWER_INDICES | RIGHT_INDICES)
    LOWER_LEFT_CORNER_INDICES = LOWER_LEFT_INDICES.difference(UPPER_INDICES | RIGHT_INDICES)

    LOWER_N_SHAPE = (LEFT_INDICES | RIGHT_INDICES | UPPER_INDICES).difference(LOWER_INDICES)
    UPPER_N_SHAPE = (LEFT_INDICES | RIGHT_INDICES | LOWER_INDICES).difference(UPPER_INDICES)
    RIGHT_N_SHAPE = (LEFT_INDICES | LOWER_INDICES | UPPER_INDICES).difference(RIGHT_INDICES)
    LEFT_N_SHAPE = (LOWER_INDICES | RIGHT_INDICES | UPPER_INDICES).difference(LEFT_INDICES)
