import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.hermite as herm
import math
import sdeint as sde
from scipy.constants import pi
import pandas as pd


def gbm(S0, mu, sigma, N):
    """Geometric Brownian motion trajectory generator"""
    t = range(N)
    S = [S0]
    for i in range(1, N):
        W = np.random.normal()
        d = (mu - 0.5 * sigma ** 2) * t[i]
        diffusion = sigma * W
        S_temp = S0 * np.exp(d + diffusion)

        if math.isinf(S_temp) or math.isnan(S_temp):
            return S[0:i]
        else:
            S.append(S_temp)
    return S


def gbm_fp(x_range, t, mu, sigma):
    """Fokker-Planck equation for gbm"""

    # PROBLEM HERE IS THAT THIS IS ONLY DEFINED FOR POSITIVE VVALUES
    muhat = mu - 0.5 * sigma ** 2
    x0 = 0.1
    pdf = [0] * len(x_range)
    # populate pdf distrbiution
    for i in range(len(x_range)):
        x = x_range[i]
        pdf[i] = (1 / (sigma * x * np.sqrt(2 * pi * t))) * np.exp(
            -((np.log(x) - np.log(x0) - muhat * t) ** 2) / (2 * t * sigma ** 2)
        )
        if math.isinf(pdf[i]) or math.isnan(pdf[i]):
            return pdf[0:i]

    return pdf
