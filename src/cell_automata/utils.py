"""
Utils for cell automata module
"""
from typing import Dict, Optional, Set, Tuple

import numpy as np


def get_balanced_random_grid(grid_width: int, grid_height: int) -> np.ndarray:
    """
    Returns a balanced randomized 2d numpy array
    :param grid_width: width in cells
    :param grid_height: height in cells
    :return: balanced random grid, balanced = the number of white and black cells are equal
    """
    grid = np.zeros((grid_width, grid_height), dtype=bool)
    grid_elements = grid_width * grid_height
    indices = np.random.permutation(grid_elements)
    grid.ravel()[indices[:grid_elements // 2]] = True
    return grid


class TargetDistanceCalculator:
    """
    Util class for calculating the distance of a numpy array from target states, using levenshtein distance
    """

    def __init__(self, grid: np.ndarray, target_grids: Dict[bool, np.ndarray]):
        """
        :param grid: grid to calculate distance for
        :param target_grids: the target grids for every value of the current cell
        """
        self._grid = grid
        self._target_neighborhoods = target_grids

    def is_on_target(self, value: Optional[bool] = None) -> bool:
        """
        is the current grid already on one of the targets
        :param value: current middle value
        """
        if value is None:
            return (np.array_equal(self._grid, self._target_neighborhoods[True]) or
                    np.array_equal(self._grid, self._target_neighborhoods[False]))

        return np.array_equal(self._grid, self._target_neighborhoods[value])

    def get_distance(self, distance_indices: Set[Tuple[int, int]]) -> Dict[bool, int]:
        """
        Returns the distance from target states for each of the different values for the cell
        :param distance_indices: indices to check the distance on
        :return: dictionary containing te distances for a True/False cell value
        """
        target_distances: Dict[bool, int] = {
            True: sum(self._target_neighborhoods[True][index] != self._grid[index] for index in distance_indices)
        }
        target_distances[False] = len(distance_indices) - target_distances[True]  # the true/false states are opposite
        return target_distances
