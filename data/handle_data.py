#Import from file
#Divide into different increments
#Output as csv?
import numpy as np
import pandas as pd
import csv


def calc_return(t, S1, S2):

	"""Given two consecutive values and the increment, calculates the return"""
	R = (252/t) * np.log(S2/S1)
	return R


def create_return_file(original_file, t, benchmark):
	"""Creates appropriate data file using appropriate holding increment"""

	dataframe = pd.read_csv(original_file, header=0, index_col=None, quoting=0)

	#see if the benchmark value impacts the 
	if benchmark == "open":

		values = dataframe["Open"]

	elif benchmark == "high":

		values = dataframe["High"]

	elif benchmark == "low":

		values = dataframe["Low"]

	elif benchmark == "close":

		values = dataframe["Close"]

	#Need to replace commas in the values
	#values = [float(values[i].replace(',', '')) for i in range(len(values))]

	#Calculate returns with 
	returns = [calc_return(t, values[i], values[i+t]) for i in range(int(len(values)-t))]

	#Write each value to csv
	with open(benchmark +"_" + str(t) + "_returns.csv", mode='w') as file:

		csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(['Returns'])


		for i in returns:
			print(i)
			csv_writer.writerow([i])




	
if __name__ == "__main__":
	create_return_file("GSPC.csv", 20, "close")