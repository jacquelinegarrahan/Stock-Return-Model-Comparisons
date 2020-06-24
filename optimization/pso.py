import numpy as np
import pandas as pd
from functools import partial

from models.gbm import gbm
from models.gbm_mod import gbm_mod
from models.qho import qho_fp


def fit_gbm_mod(X0, N, particle):
    alpha = particle[0]
    sigma_1 = particle[1]
    sigma_2 = particle[2]
    mu_step = particle[3]

    model_trajectory = gbm_mod(X0, sigma_1, sigma_2, alpha, mu_step, N)

    if len(model_trajectory) < N:
        return None

    else:
        return model_trajectory


def fit_gbm(X0, N, particle):
    mu = particle[0]
    sigma = particle[1]
    model_trajectory = gbm(X0, mu, sigma, N)

    if len(model_trajectory) < N:
        return None

    else:
        return model_trajectory


def fit_qho(X0, data, N, particle):
    min_val = min(data)
    max_val = max(data)
    x_range = np.linspace(min_val, max_val)
    C = particle[0:5]
    mw = particle[5]
    qho = qho_fp(x_range, C, mw)
    pdf = [1000 * i for i in qho]
    if len(qho) < len(x_range):
        return None

    else:
        model_trajectory = [X0]
        sampling_vector = []
        for i in range(len(pdf)):
            for j in range(int(pdf[i])):
                sampling_vector.append(x_range[i])

        for i in range(len(data)):
            model_trajectory.append(np.random.choice(sampling_vector))

        return model_trajectory


def error_calc(X0, model, data, particles):
    """
    Calculates the appropriate error for respective tests.
    
    
    """

    n = len(data)

    if model == "GBM-mod":
        f = partial(fit_gbm_mod, X0, n)
        trajectories = np.array(list(map(f, particles)))

    elif model == "GBM":
        f = partial(fit_gbm, X0, n)
        trajectories = np.array(list(map(f, particles)))

    elif model == "QHO":
        f = partial(fit_qho, X0, data, n)
        trajectories = np.array(list(map(f, particles)))

    #  error_f = partial(cramer_von_mises, data)
    error_f = partial(test_stat, data)
    errors = np.array(list(map(error_f, trajectories)))

    return errors


def cramer_von_mises(empirical, model_fit):
    """Uses the Cramér-von Mises goodness of fit statistic"""

    if not model_fit:
        return 10000000

    else:

        historical_mean = np.mean(empirical)

        M = len(model_fit)

        T = 1 / (12 * M)

        for j in range(M):
            r_j = model_fit[j] - historical_mean
            num_less = 0
            for i in range(len(empirical)):
                if empirical[i] < r_j:
                    num_less += 1

            T += (num_less / len(empirical) - (j - 0.5) / M) ** 2

        if T == "inf":
            return 1000000
        else:
            return T


def test_stat(empirical, model_fit):
    if isinstance(model_fit, (list,)) and not any([val == None for val in model_fit]):
        model_fit = np.array(model_fit)
        var_dif = abs(empirical.var() - model_fit.var())
        mean_dif = abs(empirical.mean() - model_fit.mean())

        return var_dif + mean_dif
    else:
        return 10000000
