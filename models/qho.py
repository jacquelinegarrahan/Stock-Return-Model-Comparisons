import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.hermite as herm
import math
import sdeint as sde
from scipy.constants import pi
import pandas as pd


def qho_dist(n, xs, m=1, w=1, h_bar=1):
    """
    Returns the quantum harmonic oscillator wavefunction for energy level n.

    Parameters
    ----------
    n: int
        Energy level

    xs: 
        Position coordinates

    m: float
        Mass

    w: float
        Angular frequency

    h_bar: float
        Stand-in for planck constant (functional representation doesn't require \\
        constant precision)
    
    
    """
    xmin = min(xs)
    xmax = max(xs)
    psi = []

    # coefficients for Hermite series, all 0s except the n-th term
    herm_coeff = [0] * n
    herm_coeff[-1] = 1

    for x in xs:
        psi.append(
            math.exp(-m * w * x ** 2 / (2 * h_bar))
            * herm.hermval((m * w / h_bar) ** 0.5 * x, herm_coeff)[1]
        )
    # normalization factor for the wavefunction:
    psi = np.multiply(
        psi,
        1
        / (math.pow(2, n) * math.factorial(n)) ** 0.5
        * (m * w / (pi * h_bar)) ** 0.25,
    )

    return xs, psi


def qho_fp(x_range, c, mw, h_bar=1, n=5):
    """
    Returns the quantum harmonic oscillator wavefunction for energy level n.

    Parameters
    ----------
    n: int
        Energy level

    x_range: np.ndarray
        Range of possible positions

    m: float
        Mass

    w: float
        Angular frequency

    h_bar: float
        Stand-in for planck constant (functional representation doesn't require \\
        constant precision)

    c: list
        Amplitudes of eigenstates
    
    Notes
    -----
    This is taken from original paper, "Modeling stock return distributions with a quantum harmonic oscillator"
    # Appears that t is the length of increments

    """
    pdf = []
    for i in range(len(x_range)):
        x = x_range[i]
        p = 0
        for i in range(n):
            p_prime = (
                C[i]
                * herm.hermval((mw / h_bar) ** 0.5 * x, i)
                * np.exp(-mw * x ** 2 / (h_bar))
            )
            p += p_prime.real

        if math.isinf(p) or math.isnan(p):
            return pdf[0:i]
        else:
            pdf.append(p)

    return pdf
