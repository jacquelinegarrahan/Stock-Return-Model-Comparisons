import numpy as np
import math
from scipy.constants import pi


def gbm(S0: float, mu: float, sigma: float, N: int) -> list:
    """
    Geometric Brownian motion trajectory generator

    S0: float
        starting value

    mu: float
        GMB mean

    sigma: float
        GBM variance

    N: int
        Number of steps to generate
    
    
    """
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


def gbm_fp(x_range: list, t: int, mu: float, sigma: float) -> list:
    """
    Generate pdf using Fokker-Planck equation for GBM.
    
    Parameters
    ----------
    x_range: 
        Values of x to evaluate
    
    t: int
        Time to evaluate

    mu: float
        GBM mean

    sigma: float
        GBM variance
    
    """

    # PROBLEM HERE IS THAT THIS IS ONLY DEFINED FOR POSITIVE VALUES
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
