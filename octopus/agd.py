import numpy as np

from .util import construct_g2
from .util import construct_g1
from .util import construct_g
from .util import construct_f

tau   = np.exp(1)
L     = np.exp(1)
gamma = 1

g2    = construct_g2(L, gamma, tau)
g1    = construct_g1(L, gamma, tau)
g     = construct_g(g1, g2)
nu    = - g1(2*tau) + 4 * L * tau**2
h   = 0.01 

f = construct_f(L, gamma, tau,nu, g, g1)   

def nabla_f(x):
    der_x = []
    for i in range(len(x)):
        der_x_i = f(x[:i] + [x[i]+h] + x[i+1:]) - f(x[:i] + [x[i]-h] + x[i+1:])
        der_x.append(der_x_i/(2 * h))
    return der_x  

eta = 1 / (4 * L)

def experiment_agd(x_0, iters):
    x = x_0
    d = len(x)
    values = []

    for i in range(iters):
        der_x = nabla_f(x)
    
        x_new = np.array(x) - eta * np.array(der_x)
        x     = list(x_new)
    
        values.append(f(x))
        
    return values