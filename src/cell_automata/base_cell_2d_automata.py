
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseCell2DAutomata(ABC):
    """
    Abstract class for a 2d cell automata
    """
    def __init__(self, grid_width: int, grid_height: int):
        """
        :param grid_width: automata grid width (in cells)
        :param grid_height: automata grid height (in cells)
        """
        self._grid_width = grid_width
        self._grid_height = grid_height

        self._grid: Optional[np.ndarray] = None
        self._prev_grid: Optional[np.ndarray] = None

    @abstractmethod
    def init_grid(self):
        """
        Initialize the grid
        """
        pass

    @abstractmethod
    def _inner_update_grid(self):
        """
        Update the grid according to automata rules
        """
        pass

    @abstractmethod
    def get_grid_distance(self) -> int:
        """
        Return the distance from the target states of the automata
        """
        pass

    def update_grid(self):
        """
        Updates the grid, calling the _inner_update_grid function
        """
        if self._grid is None:  # if trying to update the grid before initialize
            raise ValueError("grid has not yet been initialized")
        if self._prev_grid is not None and self.is_stable():  # if the grid is stable and no point in updating it
            return

        self._inner_update_grid()

    def is_stable(self) -> bool:
        """
        Return True iff the grid is stable
        """
        return np.array_equal(self._grid, self._prev_grid)

    def get_grid(self) -> np.ndarray:
        """
        Returns the grid 2d array
        """
        if self._grid is None:
            raise ValueError("grid has not been initialized yet")

        return self._grid.copy()

    def get_grid_shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the grid (in cells)
        """
        return self._grid_height, self._grid_width
