"""Matplotlib interface for the cellular-automata simulator."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from .base_cell_2d_automata import BaseCell2DAutomata
from .simulation import write_distance_history


class CellAutomataGUI:
    """Display and control an animated cellular-automata simulation."""

    def __init__(
        self,
        cell_automata: BaseCell2DAutomata,
        max_generations: int,
        update_interval: int,
        output_directory: str | Path = "run-results",
    ):
        self._cell_automata = cell_automata
        self._generation = 0
        self._max_generations = max_generations
        self._is_running = False
        self._history_saved = False
        self._distance_history: list[int] = []
        self._output_directory = Path(output_directory)

        self._figure, self._axes = plt.subplots()
        manager = self._figure.canvas.manager
        if manager is not None:
            manager.set_window_title("Cellular Automata Simulator")
        self._image = self._axes.imshow(
            np.zeros(shape=self._cell_automata.get_grid_shape()),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        self._generation_label = plt.figtext(0.35, 0.95, "Generation: 0")
        self._target_distance_label = plt.figtext(0.65, 0.95, "Target Distance: ~")
        self._animation = FuncAnimation(
            self._figure,
            self._update_animation,
            blit=True,
            repeat=False,
            interval=update_interval,
            cache_frame_data=False,
        )
        self._figure.canvas.mpl_connect("key_press_event", self._toggle_simulation)

        stop_button_axes = self._figure.add_axes([0.01, 0.5, 0.16, 0.05])
        self._stop_button = Button(stop_button_axes, "Stop/Continue")
        self._stop_button.on_clicked(self._toggle_simulation)
        restart_button_axes = self._figure.add_axes([0.85, 0.5, 0.1, 0.05])
        self._restart_button = Button(restart_button_axes, "Restart")
        self._restart_button.on_clicked(self._restart_simulation)

    def start(self) -> None:
        """Initialize the grid and enter Matplotlib's event loop."""
        self._initialize_run()
        plt.show(block=True)

    def _initialize_run(self) -> None:
        self._generation = 0
        self._history_saved = False
        self._cell_automata.init_grid()
        distance = self._cell_automata.get_grid_distance()
        self._distance_history = [distance]
        self._image.set_data(self._cell_automata.get_grid())
        self._generation_label.set_text("Generation: 0")
        self._target_distance_label.set_text(f"Target Distance: {distance}")
        self._is_running = True

    def _toggle_simulation(self, _event: Any) -> None:
        self._is_running = not self._is_running

    def _restart_simulation(self, _event: Any) -> None:
        self._initialize_run()
        self._figure.canvas.draw_idle()

    def _should_animate(self) -> bool:
        return self._generation < self._max_generations and not self._cell_automata.is_stable()

    def _update_animation(self, _frame: int) -> tuple[Any, ...]:
        if not self._is_running:
            return (self._image,)
        if not self._should_animate():
            self._is_running = False
            self._save_distance_history()
            return (self._image,)

        self._cell_automata.update_grid()
        distance = self._cell_automata.get_grid_distance()
        self._distance_history.append(distance)
        self._generation += 1
        self._image.set_data(self._cell_automata.get_grid())
        self._generation_label.set_text(f"Generation: {self._generation}")
        self._target_distance_label.set_text(f"Target Distance: {distance}")
        return (self._image, self._generation_label, self._target_distance_label)

    def _save_distance_history(self) -> None:
        if self._history_saved:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        write_distance_history(
            tuple(self._distance_history),
            self._output_directory / f"{timestamp}.csv",
        )
        self._history_saved = True
