# Stripe-Forming Cellular Automata

A 2024 CS MSc Computational Biology project exploring whether local cellular-automaton rules can organize a balanced random grid into alternating vertical stripes. The implementation uses wrapped neighborhoods, directional target-distance tests, an interactive Matplotlib view, and a headless path for reproducible runs.

## Tech Stack

- Python 3.10 or newer
- NumPy for grid state, wrapped neighborhoods, and target-distance calculations
- Matplotlib for the interactive simulation
- pytest and Ruff for validation and code quality
- uv for reproducible dependency and package management

## Setup

```bash
git clone https://github.com/KobieHazon/msc-computational-biology-cellular-automata.git
cd msc-computational-biology-cellular-automata
uv sync --dev
```

## Usage

Start the interactive simulator:

```bash
uv run cellular-automata
```

Use the button or any keyboard key to pause and continue; select **Restart** to generate a new balanced grid. The completed run history is written under `run-results/`.

Run the same model without a graphical display:

```bash
uv run cellular-automata --headless --seed 2024 --output run-results/reference.csv
```

The default configuration uses an `80 x 80` grid and a maximum of 250 generations. The historical runs did not record random seeds, so the regression test uses seed `2024` as a reproducible regression case; it reaches zero target distance and a stable state after 94 generations. Width must be even so the two alternating-stripe targets are well defined, and the grid must contain an even number of cells so its initial states can be balanced.

Use `--width`, `--height`, and `--generations` to explore other configurations.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The suite covers deterministic initialization, rectangular grids, target-distance behavior, CSV export, argument validation, and the complete fixed-seed regression run.

## Repository Structure

- `assignment/`: supplied exercise brief, provided as a PDF
- `src/cell_automata/`: cellular-automaton rules, simulator, GUI, and command-line interface
- `results/run-histories/`: 17 experiment histories from June 2024
- `report.pdf`: coauthored analysis and plots
- `tests/`: focused behavior and reproducibility checks

## Authorship

Solution, experiments, and report by Kobie Hazon and Daniel Ben Zion.
