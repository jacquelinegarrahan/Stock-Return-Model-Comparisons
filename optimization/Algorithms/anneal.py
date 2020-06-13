"""
Title: Opioid Simulated Annealing
    Author: William Dean
    Purpose: This file contains both a Simulated Annealing in order to
             find the global min cost for the Opioid ODE Model.
"""

import copy
import random
import time

import numpy as np
from scipy.stats import truncnorm

from Optimization.error_calculation import error_calc
import copy
import random
import time

import numpy as np
from scipy.stats import truncnorm



#  Random Functions -------------------------
# Function to Initialize a vector of Parameters.
# Continuous Initialization Functions
def randParameter(ranges):
    """ Returns a new random vector of parameters"""
    Theta = []
    for i in range(len(ranges)):
        Theta += [runif(ranges[i])]

    return Theta


def centParameter(ranges):
    """Returns a new vector parameter with middle of all ranges"""
    Theta = []
    for i in range(len(ranges)):
        Theta += [(ranges[i][0] + ranges[i][1]) / 2]
    return Theta


# Different functions to determine a new parameter
# Continuous Changes
# Uniform
def runif(range):
    """ Helper Function for runifParater to return random uniform from range low to high"""
    return np.random.uniform(range[0], range[1])


def runifParameter(parameter, i, ranges):
    """ With order of ap, cp, hr, mr, amt, ct, cmt, hor, mor, hs, ms, ds, ri, pg, fenor
        Adjusts the current ith Parameter."""
    return runif(ranges[i])


# Global scale controls the standard deviation for rnorm.
# ie Larger globscale searches a smaller area
globscale = 5


# Normal
def rnorm(parameter, range, scale=globscale):
    ''' Helper Function for rnormParameter.
        Truncated normal distribution centered around current parameter estimate.
        sd = changes with globscale parameter defined outside of function
        '''
    lower, upper = range[0], range[1]
    mu, sigma = parameter, (upper - lower) / scale
    return (truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0] - mu)


def rnormParameter(parameter, i, ranges):
    """ Returns a new ith parameter with normal distribution"""
    return rnorm(parameter=parameter, range=ranges[i])


# Neighbor Functions ---------------------
def Neighbor(Theta, ranges, n=1, rand=rnormParameter):
    """Takes in vector of parameters, Theta, and returns a new vector neighbor
       that differs in n indices.
       uses the function rand to sample next parameter
       value.
       Same Function for both continuous and discrete."""
    indices = random.sample(range(len(Theta) - 1), n)
    for i in indices:
        Theta[i] = rand(Theta[i], i, ranges=ranges)
    return Theta


#  Accept Probability for Annealing ---------------------------
def accept_propose(current, proposal, temperature):
    """Returns True if Proposal parameter vector should be accepted. else False"""
    # Proposal is better than current
    if proposal < current:
        return True
    else:
        prob = np.exp(-(proposal - current) / temperature)
        return prob > random.random()




# Search Algorithms -----------------------------------------
# Simulated Annealing ---------------------------------------
def anneal(error_calc, x0, t_range, model, data, parameter_range, t, runtime, n=25, alpha=.95, Temp=1, T_min=0.000001):
    """Takes an initial vector of Parameters and runs n iterations of Simulated Annealing"""

    # record time
    t0 = time.clock()

    # trial tracker
    trial = 1

    vbest_solution = randParameter(parameter_range)
    vbest_cost = error_calc(x0, vbest_solution, model, t_range, t, data)
    print(vbest_cost)

    # Anneal
    while time.clock() - t0 < runtime and Temp > T_min:
        # while Temp > T_min
        scale = 5
        # generate new solution and find its cost
        solution = randParameter(parameter_range)
        old_cost = error_calc(x0, solution, model, t_range, t, data)
        best_cost = old_cost
        best_solution = copy.copy(solution)
        for i in range(n):
            # Get Neighbor solution and its cost
            new_solution = copy.deepcopy(Neighbor(solution, parameter_range, rand=rnormParameter))
            new_cost = error_calc(x0, new_solution, model, t_range, t, data)

            # New Neighbor is better than previous best
            if new_cost < best_cost:
                best_cost = copy.copy(new_cost)
                best_solution = copy.deepcopy(new_solution)
                # Store if Overall Best of all Trials
                if best_cost < vbest_cost:
                    vbest_cost = copy.copy(new_cost)
                    vbest_solution = copy.deepcopy(new_solution)

            # Switch solution if needed
            if accept_propose(old_cost, new_cost, Temp):
                solution = copy.deepcopy(new_solution)
                old_cost = copy.copy(new_cost)


        Temp *= alpha
        scale *= 1.05

        # Finished Trial
        trial += 1

    print(vbest_solution)
    print(vbest_cost)
    return vbest_solution, vbest_cost