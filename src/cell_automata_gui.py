"""
GUI module for the cell automata simulator
"""
import csv
import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from cell_automata.base_cell_2d_automata import BaseCell2DAutomata


class CellAutomataGUI:
    """
    Holds properties of the GUI and controls the flow
    """
    # path to directory where the distance from target history will be saved
    AUTOMATA_TARGET_HISTORY_SAVE_DIRECTORY = Path("./automata_target_history/")

    def __init__(self,
                 cell_automata: BaseCell2DAutomata,
                 max_generations: int,
                 update_interval: int):
        """
        :param cell_automata: cell automata instance
        :param max_generations: maximum number of generations to run the automata for
        :param update_interval: interval between simulation updates
        """
        # automata simulation related properties
        self._cell_automata = cell_automata
        self._generation = 0
        self._max_generations = max_generations
        self._is_running = False
        self._automata_target_distance_history: List[int] = []
        self.AUTOMATA_TARGET_HISTORY_SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

        # GUI related properties
        self._fig, self._ax = plt.subplots()
        self._fig.canvas.set_window_title('Cell Automata Simulator')
        self._img = self._ax.imshow(np.zeros(shape=self._cell_automata.get_grid_shape()), cmap='gray', vmin=0, vmax=1)
        self._generation_label = plt.figtext(0.4, 0.95, "Generation: 0")
        self._target_distance_label = plt.figtext(0.7, 0.95, "Target Distance: ~")
        self._animation = FuncAnimation(self._fig,
                                        self._update_animation,
                                        blit=True,
                                        repeat=False,
                                        interval=update_interval)
        self._fig.canvas.mpl_connect('key_press_event', self._toggle_simulation)
        stop_button_ax = self._fig.add_axes([0.01, 0.5, 0.16, 0.05])
        self.__stop_button = Button(stop_button_ax, 'Stop/Continue')
        self.__stop_button.on_clicked(self._toggle_simulation)
        regenerate_button_ax = self._fig.add_axes([0.85, 0.5, 0.1, 0.05])
        self.__regenerate_button = plt.Button(regenerate_button_ax, 'Restart')
        self.__regenerate_button.on_clicked(self._restart_simulation)

    def start(self):
        """
        Start the cell automata simulator
        """
        self._cell_automata.init_grid()
        self._automata_target_distance_history.append(self._cell_automata.get_grid_distance())
        self._img = self._ax.imshow(self._cell_automata.get_grid(), cmap='gray', vmin=0, vmax=1)
        self._is_running = True
        plt.show(block=True)

    def _toggle_simulation(self, event):
        """
        Toggle the simulator on/off
        :param event: event that triggered the toggle
        """
        self._is_running = not self._is_running

    def _restart_simulation(self, event):
        """
        Restart the simulation
        :param event: event that triggered the toggle
        """
        self._generation = 0
        self._automata_target_distance_history = []
        self._img.set_data(np.zeros(shape=self._cell_automata.get_grid_shape()))
        self._cell_automata.init_grid()
        self._img.set_data(self._cell_automata.get_grid())
        self._generation_label.set_text(f'Generation: {self._generation}')
        self._target_distance_label.set_text('Target Distance: ~')
        plt.draw()

    def _should_animate(self) -> bool:
        """
        Should the simulation continue animating
        :return: True iff the number of generations hadn't reached max_generations and the automata is not stable
        """
        return self._generation < self._max_generations and not self._cell_automata.is_stable()

    def _update_animation(self, frame: int):
        """
        Refreshed the simulation with the next step of the automata simulation
        :param frame: number of the current frame
        """
        if not self._is_running:  # simulation stopped
            return (self._img,)
        if not self._should_animate():  # should stop animation
            self._is_running = False  # stop animation
            self._save_automata_target_distance_history()  # save the automata target distance history
            return (self._img,)

        # state updates
        self._cell_automata.update_grid()
        automata_target_distance = self._cell_automata.get_grid_distance()  # get distance from the target states
        self._automata_target_distance_history.append(automata_target_distance)
        self._generation += 1

        # GUI updates
        self._img.set_data(self._cell_automata.get_grid())
        self._generation_label.set_text(f'Generation: {self._generation}')
        self._target_distance_label.set_text(f'Target Distance: {automata_target_distance}')
        plt.show(block=False)
        return ()

    def _save_automata_target_distance_history(self):
        """
        Saves the automata distance from target states history to a csv file
        """
        automata_target_distance_history_file_name = (
                self.AUTOMATA_TARGET_HISTORY_SAVE_DIRECTORY /
                f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        )
        with open(automata_target_distance_history_file_name, 'w', newline='') as csvfile:
            distance_history_file_writer = csv.writer(csvfile)
            distance_history_file_writer.writerow(['Simulation Step', 'Minimal Target Distance'])

            for simulation_step, automata_target_distance in enumerate(self._automata_target_distance_history):
                distance_history_file_writer.writerow([simulation_step, automata_target_distance])
