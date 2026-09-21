"""Headless simulation and result export helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .stripes_cell_2d_automata import StripesCell2DAutomata


@dataclass(frozen=True)
class SimulationResult:
    """Final grid and target-distance history for one simulation."""

    grid: npt.NDArray[np.bool_]
    distances: tuple[int, ...]
    stable: bool

    @property
    def generations(self) -> int:
        return len(self.distances) - 1


def run_simulation(
    *,
    width: int = 80,
    height: int = 80,
    max_generations: int = 250,
    random_seed: int | None = None,
) -> SimulationResult:
    """Run a simulation synchronously until stable or at the generation limit."""
    if max_generations < 0:
        raise ValueError("max_generations cannot be negative")

    automata = StripesCell2DAutomata(
        width,
        height,
        random_seed=random_seed,
    )
    automata.init_grid()
    distances = [automata.get_grid_distance()]
    for _ in range(max_generations):
        automata.update_grid()
        distances.append(automata.get_grid_distance())
        if automata.is_stable():
            break

    return SimulationResult(
        grid=automata.get_grid(),
        distances=tuple(distances),
        stable=automata.is_stable(),
    )


def write_distance_history(distances: tuple[int, ...], output_path: str | Path) -> Path:
    """Write simulation target distances in the CSV format."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("Simulation Step", "Minimal Target Distance"))
        writer.writerows(enumerate(distances))
    return destination
