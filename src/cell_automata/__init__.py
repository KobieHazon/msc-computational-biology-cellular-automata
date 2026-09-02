"""Stripe-forming two-dimensional cellular automata."""

from .simulation import SimulationResult, run_simulation
from .stripes_cell_2d_automata import StripesCell2DAutomata

__all__ = ["SimulationResult", "StripesCell2DAutomata", "run_simulation"]
