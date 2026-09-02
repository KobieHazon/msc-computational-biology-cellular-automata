import csv

import numpy as np
import pytest

from cell_automata.simulation import run_simulation, write_distance_history
from cell_automata.stripes_cell_2d_automata import StripesCell2DAutomata
from cell_automata.utils import get_balanced_random_grid


def test_balanced_random_grid_has_requested_shape_and_state_counts() -> None:
    grid = get_balanced_random_grid(6, 4, np.random.RandomState(7))

    assert grid.shape == (4, 6)
    assert int(grid.sum()) == 12


def test_balanced_random_grid_is_deterministic_with_a_random_state() -> None:
    first = get_balanced_random_grid(8, 4, np.random.RandomState(42))
    second = get_balanced_random_grid(8, 4, np.random.RandomState(42))

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("width", "height", "message"),
    [
        (0, 4, "positive"),
        (5, 4, "even grid width"),
    ],
)
def test_invalid_dimensions_are_rejected(width: int, height: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StripesCell2DAutomata(width, height)


def test_balanced_random_grid_rejects_an_odd_cell_count() -> None:
    with pytest.raises(ValueError, match="even number of cells"):
        get_balanced_random_grid(3, 3)


def test_rectangular_simulation_preserves_height_width_order() -> None:
    result = run_simulation(
        width=8,
        height=4,
        max_generations=2,
        random_seed=11,
    )

    assert result.grid.shape == (4, 8)
    assert result.generations <= 2
    assert len(result.distances) == result.generations + 1


def test_perfect_stripes_have_zero_target_distance() -> None:
    automaton = StripesCell2DAutomata(6, 4)
    automaton._grid = np.tile(np.array([False, True], dtype=bool), (4, 3))

    assert automaton.get_grid_distance() == 0


def test_distance_history_uses_recovered_csv_schema(tmp_path) -> None:
    output_path = write_distance_history((8, 5, 0), tmp_path / "history.csv")

    with output_path.open(encoding="utf-8", newline="") as history_file:
        assert list(csv.reader(history_file)) == [
            ["Simulation Step", "Minimal Target Distance"],
            ["0", "8"],
            ["1", "5"],
            ["2", "0"],
        ]


def test_fixed_seed_matches_regression_baseline() -> None:
    result = run_simulation(
        width=80,
        height=80,
        max_generations=250,
        random_seed=2024,
    )

    assert result.generations == 94
    assert result.stable is True
    assert result.distances[-1] == 0
    assert {index: result.distances[index] for index in (0, 1, 10, 50, 90, 94)} == {
        0: 3188,
        1: 3133,
        10: 2703,
        50: 1229,
        90: 11,
        94: 0,
    }
