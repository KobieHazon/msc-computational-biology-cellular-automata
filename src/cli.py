"""Command-line entry point for GUI and headless cellular-automata runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .gui import CellAutomataGUI
from .simulation import run_simulation, write_distance_history
from .stripes_cell_2d_automata import StripesCell2DAutomata


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("run-results/distance-history.csv"),
    )
    args = parser.parse_args()

    if args.headless:
        result = run_simulation(
            width=args.width,
            height=args.height,
            max_generations=args.generations,
            random_seed=args.seed,
        )
        write_distance_history(result.distances, args.output)
        print(
            f"generations={result.generations} "
            f"final_distance={result.distances[-1]} stable={result.stable}"
        )
        return

    automata = StripesCell2DAutomata(
        args.width,
        args.height,
        random_seed=args.seed,
    )
    gui = CellAutomataGUI(
        automata,
        max_generations=args.generations,
        update_interval=1,
        output_directory=args.output.parent,
    )
    gui.start()


if __name__ == "__main__":
    cli()
