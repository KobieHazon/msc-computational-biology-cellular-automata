"""
Module containing the base class for a general 2d cell automata
"""
from typing import Dict, List, Set, Tuple

import numpy as np

from cell_automata.base_cell_2d_automata import BaseCell2DAutomata
from cell_automata.consts import CellNeighborhoodIndices
from cell_automata.utils import TargetDistanceCalculator, get_balanced_random_grid


class StripesCell2DAutomata(BaseCell2DAutomata):
    # target cell states
    _TARGET_CELL_STATES: Dict[bool, np.ndarray] = {
        True: np.array([
            np.array([False, True, False]),
            np.array([False, True, False]),
            np.array([False, True, False])
        ], dtype=bool),
        False: np.array([
            np.array([True, False, True]),
            np.array([True, False, True]),
            np.array([True, False, True])
        ], dtype=bool)
    }

    _STABLE_TESTS: List[Set[Tuple[int, int]]] = [
        CellNeighborhoodIndices.LEFT_N_SHAPE,
        CellNeighborhoodIndices.RIGHT_N_SHAPE,
        CellNeighborhoodIndices.UPPER_N_SHAPE,
        CellNeighborhoodIndices.LOWER_N_SHAPE
    ]

    # groups of indices to test for the new value of the cell
    # the order is important - in a state of equality of distance, the prior test will get priority
    _DISTANCE_TEST_INDICES: List[Set[Tuple[int, int]]] = [
        CellNeighborhoodIndices.LEFT_INDICES, CellNeighborhoodIndices.RIGHT_INDICES,
        CellNeighborhoodIndices.UPPER_RIGHT_CORNER_INDICES,
        CellNeighborhoodIndices.LOWER_LEFT_CORNER_INDICES,
        CellNeighborhoodIndices.UPPER_LEFT_CORNER_INDICES,
        CellNeighborhoodIndices.LOWER_RIGHT_CORNER_INDICES,
        CellNeighborhoodIndices.UPPER_INDICES, CellNeighborhoodIndices.LOWER_INDICES,
    ]

    def __init__(self, grid_width: int, grid_height: int):
        super().__init__(grid_width, grid_height)

        # target states for the entire grid according to the width and height
        self._grid_targets: List[np.ndarray] = [
            np.array([
                np.array([False, True] * (self._grid_width // 2))
                for _ in range(self._grid_height)
            ], dtype=bool),
            np.array([
                np.array([True, False] * (self._grid_width // 2))
                for _ in range(self._grid_height)
            ], dtype=bool),
        ]

    def init_grid(self):
        """
        Init the grid to a balanced (same number of black-white cells), random grid
        """
        self._grid = get_balanced_random_grid(self._grid_width, self._grid_height)
        self._prev_grid = None

    def _inner_update_grid(self):
        """
        Updates the grid with the automata rules to create vertical stripes
        """
        self._prev_grid = self._grid.copy()
        for cell_row in range(self._grid_height):
            for cell_col in range(self._grid_width):
                self._grid[cell_row, cell_col] = self._get_cell_next_value(cell_row, cell_col)

    def get_grid_distance(self) -> int:
        """
        Returns the distance of the entire grid from the target grid states
        :return: levenshtein distance of the matrix
        """

        def reduce_equal_adjacent_columns() -> np.ndarray:
            unique_cols: List[np.array] = [self._grid[:, 0]]

            for col in range(1, self._grid_width):
                if not np.array_equal(self._grid[:, col], self._grid[:, col - 1]):
                    # if the column in not equal to adjacent column
                    unique_cols.append(self._grid[:, col])

            return np.column_stack(unique_cols)

        reduced_grid = reduce_equal_adjacent_columns()
        return min(np.count_nonzero(reduced_grid != target_grid[:, :reduced_grid.shape[1]])
                   for target_grid in self._grid_targets)

    def _should_stay_stable(self, distance_calculator: TargetDistanceCalculator, curr_value: bool) -> bool:
        """
        Should the value of a cell be flipped, according to the cell's neighborhood and the tests
        :param distance_calculator: distance calculator for the cell's neighborhood
        :param curr_value: current cell value
        :return: True iff the cell should stay stable, False otherwise
        """
        if distance_calculator.is_on_target(curr_value):  # already on target state
            return True

        for stable_test in self._STABLE_TESTS:  # states for which we should not flip the cell
            if distance_calculator.get_distance(stable_test)[curr_value] == 0:
                return True

        return False

    def _get_cell_next_value(self, cell_row: int, cell_col: int) -> bool:
        """
        Get the next value for a cell according to automata rules
        :param cell_row: row of the cell
        :param cell_col: column of the cell
        :return: the new cell value
        """
        # neighborhood of the cell, with wrapping across the grid
        cell_neighborhood = np.array([
            self._prev_grid.take(cell_row - 1, mode='wrap', axis=0).take(
                [cell_col - 1, cell_col, cell_col + 1], mode='wrap'
            ),
            self._prev_grid.take(cell_row, mode='wrap', axis=0).take(
                [cell_col - 1, cell_col, cell_col + 1], mode='wrap'
            ),
            self._prev_grid.take(cell_row + 1, mode='wrap', axis=0).take(
                [cell_col - 1, cell_col, cell_col + 1], mode='wrap'
            ),
        ])
        neighborhood_target_distance_calculator = TargetDistanceCalculator(
            cell_neighborhood, self._TARGET_CELL_STATES
        )

        # if the cell should stay stable, return same value
        if self._should_stay_stable(neighborhood_target_distance_calculator, cell_neighborhood[1, 1]):
            return cell_neighborhood[1, 1]

        # value distances for the test indices
        value_test_distances: List[Dict[bool, int]] = [neighborhood_target_distance_calculator.get_distance(indices)
                                                       for indices in self._DISTANCE_TEST_INDICES]

        # choosing the test distance according to weighted random choice - disabled due to slightly lower perf
        """
        direction_weights = [min(value_for_distance.values()) for value_for_distance in value_test_distances]
        direction_to_fix = random.choices(value_test_distances, weights=direction_weights)[0]
        """
        # choosing the test distance according to the minimal test with the distance to the perfect state
        direction_to_fix = min([(value_to_distance, min(value_to_distance.values())) for value_to_distance in
                                value_test_distances], key=lambda x: x[1])[0]

        # choosing the value from the chosen test distance according to weighted choice for True/False distances
        """
        chosen_value = random.choices([True, False],
                                       weights=[direction_to_fix[False],
                                                direction_to_fix[True]],
                                       k=1)[0]
        """
        # choosing the value according to the one with the least distance to target state - slightly better
        chosen_value = min(direction_to_fix.items(), key=lambda x: x[1])[0]
        return chosen_value
