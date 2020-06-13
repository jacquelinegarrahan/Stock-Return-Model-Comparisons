#Evaluate fit
#NEED TO IMPOSE CONSTRAINTS ON THE OPTIMIZATION, make sure not to forget normalization
import numpy as np


def cramer_von_mises(empirical, model_fit):
	"""Uses the Cramér-von Mises goodness of fit statistic"""

	historical_mean = np.mean(empirical)

	M = len(model_fit)

	T = 1 / (12 * M)

	for j in range(M):
		r_j = model_fit[j]-historical_mean
		num_less = 0
		for i in range(len(empirical)):
			if empirical[i] < r_j:
				num_less +=1

		T += (num_less/len(empirical) - (j-.5)/M)**2

	return T
