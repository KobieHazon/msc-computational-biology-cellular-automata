"""
Consts for the cell automatas
"""
from typing import Set, Tuple


class CellNeighborhoodIndices:
    """
    Useful indices for calculation with neighborhood of a cell
    """
    RIGHT_INDICES: Set[Tuple[int, int]] = {(0, 2), (1, 2), (2, 2)}
    LEFT_INDICES: Set[Tuple[int, int]] = {(0, 0), (1, 0), (2, 0)}
    UPPER_INDICES: Set[Tuple[int, int]] = {(0, 0), (0, 1), (0, 2)}
    LOWER_INDICES: Set[Tuple[int, int]] = {(2, 0), (2, 1), (2, 2)}

    UPPER_RIGHT_INDICES: Set[Tuple[int, int]] = UPPER_INDICES | RIGHT_INDICES
    LOWER_RIGHT_INDICES: Set[Tuple[int, int]] = LOWER_INDICES | RIGHT_INDICES
    UPPER_LEFT_INDICES: Set[Tuple[int, int]] = UPPER_INDICES | LEFT_INDICES
    LOWER_LEFT_INDICES: Set[Tuple[int, int]] = LOWER_INDICES | LEFT_INDICES

    UPPER_RIGHT_CORNER_INDICES: Set[Tuple[int, int]] = UPPER_RIGHT_INDICES.difference(LEFT_INDICES | LOWER_INDICES)
    LOWER_RIGHT_CORNER_INDICES: Set[Tuple[int, int]] = LOWER_RIGHT_INDICES.difference(UPPER_INDICES | LEFT_INDICES)
    UPPER_LEFT_CORNER_INDICES: Set[Tuple[int, int]] = UPPER_LEFT_INDICES.difference(LOWER_INDICES | RIGHT_INDICES)
    LOWER_LEFT_CORNER_INDICES: Set[Tuple[int, int]] = LOWER_LEFT_INDICES.difference(UPPER_INDICES | RIGHT_INDICES)

    LOWER_N_SHAPE = (LEFT_INDICES | RIGHT_INDICES | UPPER_INDICES).difference(LOWER_INDICES)
    UPPER_N_SHAPE = (LEFT_INDICES | RIGHT_INDICES | LOWER_INDICES).difference(UPPER_INDICES)
    RIGHT_N_SHAPE = (LEFT_INDICES | LOWER_INDICES | UPPER_INDICES).difference(RIGHT_INDICES)
    LEFT_N_SHAPE = (LOWER_INDICES | RIGHT_INDICES | UPPER_INDICES).difference(LEFT_INDICES)
