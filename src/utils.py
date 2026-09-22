"""Shared utilities for cellular-automata implementations."""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np
import numpy.typing as npt

from .consts import CellIndex

RandomGenerator = np.random.Generator | np.random.RandomState


def get_balanced_random_grid(
    grid_width: int,
    grid_height: int,
    random_generator: RandomGenerator | None = None,
) -> npt.NDArray[np.bool_]:
    """Return a random grid containing equal numbers of both cell states."""
    grid_elements = grid_width * grid_height
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError("grid dimensions must be positive")
    if grid_elements % 2:
        raise ValueError("a balanced grid requires an even number of cells")

    grid = np.zeros((grid_height, grid_width), dtype=bool)
    if random_generator is None:
        indices = np.random.permutation(grid_elements)
    else:
        indices = random_generator.permutation(grid_elements)
    grid.ravel()[indices[: grid_elements // 2]] = True
    return grid


class TargetDistanceCalculator:
    """Calculate Hamming distances from neighborhood target states."""

    def __init__(
        self,
        grid: npt.NDArray[np.bool_],
        target_grids: Mapping[bool, npt.NDArray[np.bool_]],
    ):
        self._grid = grid
        self._target_neighborhoods = target_grids

    def is_on_target(self, value: bool | None = None) -> bool:
        """Return whether the neighborhood matches one requested target."""
        if value is None:
            return any(
                np.array_equal(self._grid, target) for target in self._target_neighborhoods.values()
            )
        return np.array_equal(self._grid, self._target_neighborhoods[value])

    def get_distance(self, distance_indices: Collection[CellIndex]) -> dict[bool, int]:
        """Return target-state mismatch counts over selected cells."""
        true_distance = sum(
            bool(self._target_neighborhoods[True][index] != self._grid[index])
            for index in distance_indices
        )
        return {
            True: true_distance,
            False: len(distance_indices) - true_distance,
        }
