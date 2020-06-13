import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.hermite as herm
import math
import sdeint as sde
from scipy.constants import pi
import pandas as pd

#Notes:
#Inverse sampling from the QHO may be problematic.


def gbm(S0, mu, sigma, N):
    """Geometric Brownian motion trajectory generator"""
    t = range(N)
    S = [S0]
    for i in range(1, N):
        W = np.random.normal()
        d = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W
        S_temp = S0 * np.exp(d + diffusion)

        if math.isinf(S_temp) or math.isnan(S_temp):
            return S[0:i]
        else:
            S.append(S_temp)
    return S


def gbm_fp(x_range, t, mu, sigma):
    """Fokker-Planck equation for gbm"""

    #PROBLEM HERE IS THAT THIS IS ONLY DEFINED FOR POSITIVE VVALUES
    muhat = mu - 0.5*sigma**2
    x0 = 0.1
    pdf = [0]*len(x_range)
    #populate pdf distrbiution
    for i in range(len(x_range)):
        x = x_range[i]
        pdf[i] = (1/(sigma *x *np.sqrt(2*pi*t))) * np.exp(-(np.log(x)- \
            np.log(x0)-muhat*t)**2/(2*t*sigma**2))
        if math.isinf(pdf[i]) or math.isnan(pdf[i]):
            return pdf[0:i]

    return pdf


def gbm_mod(X0, sigma_1, sigma_2, alpha, mu_step, N):
    """modified GBM with Ornstein-Uhlenbeck term"""
    #mu_step is the number of previous steps to include in the mean
    t = np.linspace(0,1,N+1)
    mus= [0]
    X = [X0]
    mus[0] = X0
    for i in range(N-1):
        if i < mu_step:
            mu = np.average(X[0:i+1])
        else:
            mu = np.average(X[int(i-mu_step):int(i+1)])

        mus.append(mu)

        B = np.random.normal()

        Y = X0 * np.exp(alpha*mu*i - 0.5*i*sigma_1**2 + sigma_1 * B)

        #Use the sde library to complete integrals in second part of equation
        increments = 1000
        tspan = np.linspace(0, i, increments)

        def f(x, t):
            return (-alpha-sigma_1*sigma_2)/(X0 * np.exp((alpha*mu - 0.5*sigma_1**2)*t + sigma_1 * B))

        def g(x,t):
            return sigma_2/(X0 * np.exp((alpha*mu - 0.5*sigma_1**2)*t + sigma_1 * B))

        sde_int = sde.itoint(f,g, X0, tspan)

        #NEED TO ADD QUIT CONDITION IF INFINITY
        x = Y * (X0 + np.sum(sde_int))
        if math.isinf(x) or math.isnan(x):
            return X[0:i]
        else:
            print(x)
            X.append(x)

    return X


#def gbm_mod_fp():


def quantum_harmonic_oscillator(n, xs, m=1, w=1, h_bar=1):
    """Will return the quantum harmonic oscillator wavefunction for energy level n. NOT THE TRAJECTORY"""
    xmin = min(xs)
    xmax= max(xs)
    psi = []
    # coefficients for Hermite series, all 0s except the n-th term
    herm_coeff = [0]*n
    herm_coeff[-1] =1

    for x in xs:
        psi.append(math.exp(-m*w*x**2/(2*h_bar)) * herm.hermval((m*w/h_bar)**0.5 * x, herm_coeff)[1])
    # normalization factor for the wavefunction:
    psi = np.multiply(psi, 1 / (math.pow(2, n) * math.factorial(n))**0.5 * (m*w/(pi*h_bar))**0.25)

    return xs, psi


def qho_fp(x_range, t, C, mw, h_bar=1, n=5):
    #This is taken from original paper, "Modeling stock return distributions with a quantum harmonic oscillator"
    # Appears that t is the length of increments
    pdf = []
    for i in range(len(x_range)):
        x = x_range[i]
        p = 0
        for i in range(n):
        #	En = n*h_bar*w
        #	p += (An / np.sqrt((s**n)*np.factorial(n))) * np.sqrt((m*w)/(pi*h_bar))*np.exp(-En*t) * \
        #	hermval(np.sqrt(m*w/h_bar)*x) * np.exp(-m*w*x**2/h_bar)
            #Using the simplified fit formula from the paper
            p_prime = C[i] * herm.hermval((mw/h_bar)**0.5 * x, i) * np.exp(-mw*x**2/(h_bar))
            p += p_prime.real

        if math.isinf(p) or math.isnan(p):
            return pdf[0:i]
        else:
            pdf.append(p)

    return pdf



def fit_gbm_mod(X0, parameters, N):
    alpha = parameters[0]
    sigma_1 = parameters[1]
    sigma_2 = parameters[2]
    mu_step = parameters[3]

    model = gbm_mod(X0, sigma_1, sigma_2, alpha, mu_step, N)
    print(model)
    print(len(model))
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



if __name__ == "__main__":

    t=20



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

    N = len(test_data)
    parameters = [0.2, 0.2, 0.086, 0.182, 0.133, 0.928]
    print(test_data[0])

    X = fit_qho(test_data[0], parameters, t, test_data, 100)
    x = range(len(X))
    plt.plot(x, X)
    plt.xlim([0,100])
    plt.show()

