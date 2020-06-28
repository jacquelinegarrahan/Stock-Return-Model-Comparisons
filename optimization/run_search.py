import numpy as np
import pandas as pd
import pyswarms as ps
from functools import partial
from pyswarms.utils.search import RandomSearch
from pyswarms.utils.plotters import plot_cost_history
from optimization.pso import error_calc

PARAMETER_RANGES = {
    "QHO": {
        "C0": [0, 0.2],
        "C1": [0, 0.2],
        "C2": [0, 0.2],
        "C3": [0, 0.2],
        "C4": [0, 0.2],
        "C5": [0, 0.2],
        "mw": [0, 1],
    },
    "GBM": {"mu": [-0.1, 0.1], "sigma": [-0.5, 0.5],},
    "GBM-mod": {
        "alpha": [0, 0.2],
        "sigma1": [0, 0.2],
        "sigma2": [0, 0.2],
        "mu_time": [0, 100],
    },
}


def run_search(
    data: np.ndarray, parameter_ranges: dict, model: str, search: str = "pso"
):
    """
    Launch search using PSO or Genetic search.

    Parameters
    ----------
    data: np.ndarray
        Empirical data

    parameter_ranges: dict
        Ranges to search over

    model: str
        Which model to run

    search: str
        Indicates what type of search to run
    """
    X0 = data[0]

    if search == "pso":
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

        fx = partial(error_calc, X0, model, data)
        optimizer = ps.single.GlobalBestPSO(
            n_particles=10,
            dimensions=len(parameter_ranges),
            options=options,  # bounds=parameter_ranges
        )

        # Perform optimization
        cost, pos = optimizer.optimize(fx, iters=1000)

        plot_cost_history(optimizer.cost_history)

    elif search == "genetic":
        pass

    return cost, pos
