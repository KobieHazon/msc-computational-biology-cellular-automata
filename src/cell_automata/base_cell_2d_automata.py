"""Abstract interface for two-dimensional cellular automata."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class BaseCell2DAutomata(ABC):
    """Base class for a finite two-dimensional cellular automaton."""

    def __init__(self, grid_width: int, grid_height: int):
        if grid_width <= 0 or grid_height <= 0:
            raise ValueError("grid dimensions must be positive")
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._grid: npt.NDArray[np.bool_] | None = None
        self._prev_grid: npt.NDArray[np.bool_] | None = None

    @abstractmethod
    def init_grid(self) -> None:
        """Initialize the automaton's grid."""

    @abstractmethod
    def _inner_update_grid(self) -> None:
        """Apply one generation of automaton rules."""

    @abstractmethod
    def get_grid_distance(self) -> int:
        """Return the current distance from the automaton's target state."""

    def update_grid(self) -> None:
        """Advance the initialized grid by one generation."""
        if self._grid is None:
            raise ValueError("grid has not yet been initialized")
        if self._prev_grid is not None and self.is_stable():
            return
        self._inner_update_grid()

    def is_stable(self) -> bool:
        """Return whether the latest update left the grid unchanged."""
        return self._prev_grid is not None and np.array_equal(self._grid, self._prev_grid)

    def get_grid(self) -> npt.NDArray[np.bool_]:
        """Return a defensive copy of the current grid."""
        if self._grid is None:
            raise ValueError("grid has not been initialized yet")
        return self._grid.copy()

    def get_grid_shape(self) -> tuple[int, int]:
        """Return grid shape as `(height, width)`."""
        return self._grid_height, self._grid_width
