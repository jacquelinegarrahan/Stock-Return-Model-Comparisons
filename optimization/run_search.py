"""Runs optimization search for model parameters"""
import xlsxwriter
from Optimization.error_calculation import error_calc
import numpy as np
import pandas as pd
from Optimization.Algorithms.particle_swarm import particle_swarm
from Optimization.Algorithms.anneal import anneal

parameter_ranges = {"QHO": [[0, 0.2], [0, 0.2], [0, 0.2], [0, 0.2], [0, 0.2],  [0, 1]],
					"GBM-mod": [[0, 0.2], [0, 0.2], [0, 0.2], [0, 100]],
					"GBM": [[-0.1, 0.1], [-0.5,  0.5]]}
models = ["QHO"]


def run_search(algorithm, error_calc, models, parameter_ranges, t, runtime):
	"""Runs search, imports test data,  and writes to xlsx file"""

	#import correct data file
	if t == 1:
		file = "~/projects/mathematical_modeling/term_project/Data/close_1_returns.csv"
		dataframe = pd.read_csv(file, header=0, index_col=None, quoting=0)
		test_data = dataframe["Returns"]
		t_range = range(len(test_data))

	elif t == 5:
		file = "~/projects/mathematical_modeling/term_project/Data/close_5_returns.csv"
		dataframe = pd.read_csv(file, header=0, index_col=None, quoting=0)
		test_data = dataframe["Returns"]
		t_range = range(len(test_data))

	elif t == 20:
		file = "~/projects/mathematical_modeling/term_project/Data/close_20_returns.csv"
		dataframe = pd.read_csv(file, header=0, index_col=None, quoting=0)
		test_data = dataframe["Returns"]
		t_range = range(len(test_data))

	else:
		print("t does not exist")


	workbook = xlsxwriter.Workbook("param_search_results_" + str(t) + "_GHO_" + algorithm.__name__ + ".xlsx", {'nan_inf_to_errors': True})
	worksheet = workbook.add_worksheet()
	row = 0
	col = 0
	X0 = test_data[0]


	for i in range(len(models)):
		model = models[i]
		if models[i] == "GBM":
			header = ["GBM", "error", "mu", "sigma"]
		elif models[i] == "GBM-mod":
			header = ["GBM-mod", "error", "alpha", "sigma1", "sigma2", "mu_time"]
		elif models[i] == "QHO":
			header = ["QHO", "error", "C0", "C1", "C2", "C3", "C4", "C5", "mw"]
		else:
			print("Error in run_search model naming")

		#Populate header for each model 
		for i in range(len(header)):
			worksheet.write(row, col+i, header[i])


		row += 1

		#Run model then populate worksheet
		parameters, error = algorithm(error_calc, X0, t_range, model, test_data, parameter_ranges[model], t, runtime)

		entry = [""]+[error] + parameters

		for j in range(len(entry)):
			worksheet.write(row, col+j, entry[j])

		row +=1

	print("Search is complete.")
	workbook.close()


t=5
runtime = 600
algorithm = particle_swarm


run_search(algorithm, error_calc, models, parameter_ranges, t, runtime)
