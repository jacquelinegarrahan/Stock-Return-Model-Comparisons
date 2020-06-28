import numpy as np
import math
import sdeint as sde
import pandas as pd


def gbm_mod(
    X0: float, sigma_1: float, sigma_2: float, alpha: float, mu_step: int, N: int
) -> list:
    """
    Generates modified GBM with Ornstein-Uhlenbeck term

    Parameters
    ----------
    X0: float

    sigma_1: float

    sigma_2: float

    alpha: float

    mu_step: int
        Steps to evaluate moving mean

    N: int
        Number of steps to generate
    
    """
    # mu_step is the number of previous steps to include in the mean
    t = np.linspace(0, 1, N + 1)
    mus = [0]
    X = [X0]
    mus[0] = X0
    for i in range(N - 1):
        if i < mu_step:
            mu = np.average(X[0 : i + 1])
        else:
            mu = np.average(X[int(i - mu_step) : int(i + 1)])

        mus.append(mu)

        B = np.random.normal()

        Y = X0 * np.exp(alpha * mu * i - 0.5 * i * sigma_1 ** 2 + sigma_1 * B)

        # Use the sde library to complete integrals in second part of equation
        increments = 1000
        tspan = np.linspace(0, i, increments)

        def f(x, t):
            return (-alpha - sigma_1 * sigma_2) / (
                X0 * np.exp((alpha * mu - 0.5 * sigma_1 ** 2) * t + sigma_1 * B)
            )

        def g(x, t):
            return sigma_2 / (
                X0 * np.exp((alpha * mu - 0.5 * sigma_1 ** 2) * t + sigma_1 * B)
            )

        sde_int = sde.itoint(f, g, X0, tspan)

        # NEED TO ADD QUIT CONDITION IF INFINITY
        x = Y * (X0 + np.sum(sde_int))
        if math.isinf(x) or math.isnan(x):
            return X[0:i]
        else:
            X.append(x)

    return X
