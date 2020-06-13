# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:37:43 2017

@author: jgarrahan
"""
import time
import numpy as np


def parameter_bounds(parameter_range):
    lower_bound = []
    upper_bound = []
    for i in range(len(parameter_range)):
        lower_bound.append(parameter_range[i][0])
        upper_bound.append(parameter_range[i][1])
    return lower_bound, upper_bound


def mutate(parameters, mu, lower_bound, upper_bound):
    """
    :type parameters: object
    """
    max_mutations = int(np.ceil(mu*len(parameters)))
    nVar = len(parameters)
    num_mutate = np.random.choice((range(max_mutations)))
    indices = np.random.choice(nVar, num_mutate, replace=False)

    xnew = parameters
    for j in indices:
        xnew[j] += np.random.uniform(lower_bound[j], upper_bound[j])
    for x in range(0, len(xnew)):
        xnew[x] = max(xnew[x], lower_bound[x])
        xnew[x] = min(xnew[x], upper_bound[x])
    return xnew


# define particle attributes
class Particle:
    def __init__(self):
        self.parameters = []
        self.error = []
        self.velocity = []
        self.parameters_best = []
        self.error_best = []


def particle_swarm(error_calc, x0, t_range, model, test_data, parameter_range, t, runtime):
    omega=1
    swarm_size = 10
    max_iter = 100
    mu = 0.2
    particle_search_scalar=2
    global_search_scalar=2
    lower_bound, upper_bound = parameter_bounds(parameter_range)
    # Define velocity limits
    Vmax = 0.3 * np.subtract(upper_bound, lower_bound)
    Vmin = -Vmax

    print('inside particle swarm')

    # Initialize particles
    particle = np.empty(shape=(swarm_size,), dtype=object)
    global_parameters = np.ndarray.tolist(np.zeros(len(lower_bound)))
    global_error = float('inf')

    for i in range(swarm_size):
        particle[i] = Particle()
        particle[i].parameters = [np.random.uniform(lower_bound[j], upper_bound[j]) for j in range(len(lower_bound))]
        particle[i].error = error_calc(x0, particle[i].parameters, model, t_range, t, test_data)
        # update remaining attributes
        particle[i].velocity = np.zeros(len(lower_bound))
        particle[i].parameters_best = particle[i].parameters
        particle[i].error_best = particle[i].error

    timelist = []
    least_error_tracker = []
    it = 1
    start = time.clock()

    while time.clock() - start < runtime and it < max_iter:
        print(time.clock()-start)
        for i in range(swarm_size):
            # random array for local particle
            rp = np.random.rand(len(lower_bound))
            # random array for global particle
            rg = np.random.rand(len(lower_bound))
            # difference between particle best and current
            difference1 = np.subtract(particle[i].parameters_best, particle[i].parameters)
            # difference between particle current and global best
            difference2 = np.subtract(global_parameters, particle[i].parameters)
            # modify velocity using weighted differences
            particle[i].velocity = omega * particle[i].velocity + np.dot(particle_search_scalar * rp,
                                                                         difference1) + np.dot(
                global_search_scalar * rg, difference2)

            # iterate over parameters and velocity to update
            for j in range(len(lower_bound)):
                # update velocity
                particle[i].velocity[j] = max(particle[i].velocity[j], Vmin[j])
                particle[i].velocity[j] = min(particle[i].velocity[j], Vmax[j])
                # update parameters
                particle[i].parameters[j] = particle[i].parameters[j] + particle[i].velocity[j]

                if particle[i].parameters[j] < lower_bound[j] or particle[i].parameters[j] > upper_bound[j]:
                    particle[i].velocity[j] = -particle[i].velocity[j]

                particle[i].parameters[j] = max(particle[i].parameters[j], lower_bound[j])
                particle[i].parameters[j] = min(particle[i].parameters[j], upper_bound[j])

            # update solution and error


            particle[i].error = error_calc(x0, particle[i].parameters,  model, t_range, t, test_data)

            # update particle best
            if particle[i].error < particle[i].error_best:
                particle[i].error_best = particle[i].error
                for j in range(len(particle[i].parameters_best)):
                    particle[i].parameters_best[j] = particle[i].parameters[j]

                if particle[i].error_best < global_error:
                    global_error = particle[i].error_best
                    for j in range(len(global_parameters)):
                        global_parameters[j] = particle[i].parameters_best[j]

            # incorporate mutation for more robust results
            new_particle = Particle()
            new_particle.parameters = mutate(particle[i].parameters, mu, lower_bound, upper_bound)
            new_particle.error = error_calc(x0, new_particle.parameters,  model, t_range, t, test_data)

            if new_particle.error < particle[i].error:
                particle[i].error = new_particle.error
                for j in range(len(particle[i].parameters)):
                    particle[i].parameters[j] = new_particle.parameters[j]

                if particle[i].error < particle[i].error_best:
                    particle[i].error_best = particle[i].error
                    for j in range(len(particle[i].parameters_best)):
                        particle[i].parameters_best[j] = particle[i].parameters[j]

                    # update global best particle
                    if particle[i].error_best < global_error:
                        global_error = particle[i].error_best
                        for j in range(len(global_parameters)):
                            global_parameters[j] = particle[i].parameters_best[j]

        it += 1
        stop = time.clock()
        timelist.append(stop - start)
        least_error_tracker.append(global_error)

    print(global_parameters)
    print(global_error)

    return global_parameters, global_error

