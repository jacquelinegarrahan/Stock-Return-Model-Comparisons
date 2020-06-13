"""Error calculation script for Mathematical Modeling term project"""
import importlib.util

spec1 = importlib.util.spec_from_file_location("models", "~/project/mathematical_modeling/term_project/models.py")
spec2 = importlib.util.spec_from_file_location("fit_stats", "~/project/mathematical_modeling/term_project/fit_stats.py")
models = importlib.util.module_from_spec(spec1)
fit_stats = importlib.util.module_from_spec(spec1)
from models import gbm_fp, gbm_mod, qho_fp, gbm
from fit_stats import cramer_von_mises
import numpy as np
import pandas as pd


def fit_gbm_mod(X0, parameters, N):
    alpha = parameters[0]
    sigma_1 = parameters[1]
    sigma_2 = parameters[2]
    mu_step = parameters[3]

    model = gbm_mod(X0, sigma_1, sigma_2, alpha, mu_step, N)

    if len(model) < N:
        return None

    else:
        return model


def fit_gbm(X0, parameters, N):
    mu = parameters[0]
    sigma = parameters[1]
    model = gbm(X0, mu, sigma, N)

    if len(model) < N:
        return None

    else:
        return model


def fit_qho(X0, parameters, t, data, N):
    min_val = min(data)
    max_val = max(data)
    x_range = np.linspace(min_val, max_val)
    qho = qho_fp(x_range, t, parameters[0:5], parameters[5])
    pdf = [1000 * i for i in qho]
    if len(qho) < len(x_range):
        return None

    else:
        model = [X0]
        sampling_vector = []
        for i in range(len(pdf)):
            for j in range(int(pdf[i])):
                sampling_vector.append(x_range[i])

        for i in range(len(data)):
            model.append(np.random.choice(sampling_vector))

        return model


def error_calc(X0, parameters, model, t_range, t, data):
    """Calculates the appropriate error for respective tests"""

    if model == "GBM-mod":

        model = fit_gbm_mod(X0, parameters, len(data))

        if model == None:

            error_val = False

        else:

            error_val = True

    elif model == "GBM":

        #	density = fit_gbm(X0, parameters, len(data))
        model = gbm(X0, parameters[0], parameters[1], len(data))

        if model == None:

            error_val = False

        else:

            error_val = True

    elif model == "QHO":

        model = fit_qho(X0, parameters, t, data, len(data))

        if model == None:

            error_val = False

        else:

            error_val = True

    if error_val == False:
        # Set T very large so the parameters rejected
        T = 100000000

    else:
        T = cramer_von_mises(data, model)
        if T == "inf":
            T = 100000000

    return T


if __name__ == "__main__":
    file = "~/projects/mathematical_modeling/term_project/Data/close_1_returns.csv"
    dataframe = pd.read_csv(file, header=0, index_col=None, quoting=0)
    parameters = [0.01, 0.01, 0.01, 25]
    model = "GBM-mod"
    print(error_calc(parameters, model, np.linspace(-10, 10, 50), 1, dataframe["Returns"]))
