import matplotlib.pyplot as plt
import numpy as np

T = 100
mu = 0.15
sigma = 0.5
S0 = 1
dt = 0.01
N = round(T/dt)
t = np.linspace(0, T, N)
W = np.random.standard_normal(size = N)
print(W)
W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
X = (mu-0.5*sigma**2)*t + sigma*W
S = S0*np.exp(X) ### geometric brownian motion ###
plt.plot(t, S)
plt.show()