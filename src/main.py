"""
Main module for running the vertical stripes cell automata simulation
"""
from cell_automata.stripes_cell_2d_automata import StripesCell2DAutomata
from cell_automata_gui import CellAutomataGUI

GRID_SIZE = (80, 80)  # rows and columns of the automata grid
MAX_GENERATIONS = 250  # maximum number of generations to run the automata for
UPDATE_INTERVAL = 1  # interval of screen refreshes, in ms


def run_automata():
    """
    Runs the vertical stripes cell automata simulation
    """
    stripes_cell_automata = StripesCell2DAutomata(GRID_SIZE[0], GRID_SIZE[1])
    cell_automata_gui = CellAutomataGUI(stripes_cell_automata, MAX_GENERATIONS, UPDATE_INTERVAL)
    cell_automata_gui.start()


if __name__ == "__main__":
    run_automata()
