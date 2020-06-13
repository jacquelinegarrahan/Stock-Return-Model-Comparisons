#!/usr/bin/env python3
""""

=====================================
Quantum harmonic oscillator for QuantMechpy
=====================================
by Jacqueline Garrahan

"""
from __future__ import division
from numpy.polynomial.hermite import *
import numpy as np
import pylab
import math
from scipy.constants import pi


def quantum_harmonic_oscillator(n, xs, m=1, w=1, h_bar=1):
    """Will return the quantum harmonic oscillator for energy level n"""
    xmin = min(xs)
    xmax= max(xs)
    psi = []
    # coefficients for Hermite series, all 0s except the n-th term
    herm_coeff = [0]*n
    herm_coeff[-1] =1

    for x in xs:
        psi.append(math.exp(-m*w*x**2/(2*h_bar)) * hermval((m*w/h_bar)**0.5 * x, herm_coeff))
    # normalization factor for the wavefunction:
    psi = np.multiply(psi, 1 / (math.pow(2, n) * math.factorial(n))**0.5 * (m*w/(pi*h_bar))**0.25)

    return xs, psi




##IN ORDER TO SAMPLE, will have to use inverse transform sampling



#update quantmechpy with this
n=2
xs, psi = quantum_harmonic_oscillator(2)
pylab.plot(xs, psi)
pylab.xlabel("$x$", size=18)
pylab.ylabel("$\psi_{" + str(n) + "}(x)$", size=18)
pylab.title("Quantum Harmonic Oscillator Wavefunction ($n = " + str(n) + "$)", size=14)
pylab.savefig("QHOn=" + str(n) + ".png",bbox_inches="tight",dpi=600)
pylab.show()