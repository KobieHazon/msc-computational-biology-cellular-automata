"""Cellular automaton that converges toward alternating vertical stripes."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from .base_cell_2d_automata import BaseCell2DAutomata
from .consts import CellIndex, CellNeighborhoodIndices
from .utils import TargetDistanceCalculator, get_balanced_random_grid


class StripesCell2DAutomata(BaseCell2DAutomata):
    """Apply local rules favoring synchronized alternating columns."""

    _TARGET_CELL_STATES: ClassVar[Mapping[bool, npt.NDArray[np.bool_]]] = {
        True: np.array(
            [[False, True, False], [False, True, False], [False, True, False]],
            dtype=bool,
        ),
        False: np.array(
            [[True, False, True], [True, False, True], [True, False, True]],
            dtype=bool,
        ),
    }

    _STABLE_TESTS: ClassVar[tuple[Collection[CellIndex], ...]] = (
        CellNeighborhoodIndices.LEFT_N_SHAPE,
        CellNeighborhoodIndices.RIGHT_N_SHAPE,
        CellNeighborhoodIndices.UPPER_N_SHAPE,
        CellNeighborhoodIndices.LOWER_N_SHAPE,
    )

    # Tie order is part of the rule: horizontal directions are preferred.
    _DISTANCE_TEST_INDICES: ClassVar[tuple[Collection[CellIndex], ...]] = (
        CellNeighborhoodIndices.LEFT_INDICES,
        CellNeighborhoodIndices.RIGHT_INDICES,
        CellNeighborhoodIndices.UPPER_RIGHT_CORNER_INDICES,
        CellNeighborhoodIndices.LOWER_LEFT_CORNER_INDICES,
        CellNeighborhoodIndices.UPPER_LEFT_CORNER_INDICES,
        CellNeighborhoodIndices.LOWER_RIGHT_CORNER_INDICES,
        CellNeighborhoodIndices.UPPER_INDICES,
        CellNeighborhoodIndices.LOWER_INDICES,
    )

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        *,
        random_seed: int | None = None,
    ):
        super().__init__(grid_width, grid_height)
        if grid_width % 2:
            raise ValueError("alternating stripe targets require an even grid width")
        if (grid_width * grid_height) % 2:
            raise ValueError("a balanced initial grid requires an even cell count")

        target_row = np.tile(np.array([False, True], dtype=bool), grid_width // 2)
        first_target = np.tile(target_row, (grid_height, 1))
        self._grid_targets = (first_target, np.logical_not(first_target))
        self._random_generator = (
            np.random.RandomState(random_seed) if random_seed is not None else None
        )

    def init_grid(self) -> None:
        """Initialize a random grid with equal black and white cell counts."""
        self._grid = get_balanced_random_grid(
            self._grid_width,
            self._grid_height,
            self._random_generator,
        )
        self._prev_grid = None

    def _inner_update_grid(self) -> None:
        assert self._grid is not None
        self._prev_grid = self._grid.copy()
        for cell_row in range(self._grid_height):
            for cell_col in range(self._grid_width):
                self._grid[cell_row, cell_col] = self._get_cell_next_value(cell_row, cell_col)

    def get_grid_distance(self) -> int:
        """Return minimum target mismatch after collapsing equal adjacent columns."""
        if self._grid is None:
            raise ValueError("grid has not been initialized yet")

        unique_columns = [self._grid[:, 0]]
        for column in range(1, self._grid_width):
            if not np.array_equal(self._grid[:, column], self._grid[:, column - 1]):
                unique_columns.append(self._grid[:, column])
        reduced_grid = np.column_stack(unique_columns)

        return min(
            int(np.count_nonzero(reduced_grid != target[:, : reduced_grid.shape[1]]))
            for target in self._grid_targets
        )

    def _should_stay_stable(
        self,
        distance_calculator: TargetDistanceCalculator,
        current_value: bool,
    ) -> bool:
        if distance_calculator.is_on_target(current_value):
            return True
        return any(
            distance_calculator.get_distance(test)[current_value] == 0
            for test in self._STABLE_TESTS
        )

    def _get_cell_next_value(self, cell_row: int, cell_col: int) -> bool:
        assert self._prev_grid is not None
        neighborhood = np.array(
            [
                self._prev_grid.take(cell_row - 1, mode="wrap", axis=0).take(
                    [cell_col - 1, cell_col, cell_col + 1], mode="wrap"
                ),
                self._prev_grid.take(cell_row, mode="wrap", axis=0).take(
                    [cell_col - 1, cell_col, cell_col + 1], mode="wrap"
                ),
                self._prev_grid.take(cell_row + 1, mode="wrap", axis=0).take(
                    [cell_col - 1, cell_col, cell_col + 1], mode="wrap"
                ),
            ]
        )
        calculator = TargetDistanceCalculator(neighborhood, self._TARGET_CELL_STATES)
        current_value = bool(neighborhood[1, 1])
        if self._should_stay_stable(calculator, current_value):
            return current_value

        value_test_distances = [
            calculator.get_distance(indices) for indices in self._DISTANCE_TEST_INDICES
        ]
        direction_to_fix = min(
            value_test_distances,
            key=lambda value_to_distance: min(value_to_distance.values()),
        )
        return min(direction_to_fix, key=direction_to_fix.get)
