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
   

def get_rand_vec(dims):
    x = np.random.standard_normal(dims)
    return x / np.linalg.norm(x)

t_thresh  = -1  
t_noise   = - t_thresh -1 
g_thresh  = np.exp(1) * gamma / 100
r         = np.exp(1) / 100

def nabla_f(x):
    der_x = []
    for i in range(len(x)):
        der_x_i = f(x[:i] + [x[i]+h] + x[i+1:]) - f(x[:i] + [x[i]-h] + x[i+1:])
        der_x.append(der_x_i/(2 * h))
    return der_x

eta = 1 / (4 * L)

def experiment_pagd(x_0, iters):
    x = x_0
    d = len(x)
    t_noise   = - t_thresh -1
    values = []

    for i in range(iters):
        der_x = nabla_f(x)
        der_x = np.array(der_x)
    
        if (3/4) * g_thresh >= np.linalg.norm(der_x) and  (i - t_noise > t_thresh):
            noise   = r * get_rand_vec(len(x))
            x       = np.array(x) + noise
            x       = list(x)
            der_x   = nabla_f(x)
            der_x   = np.array(der_x)
            t_noise = i   
    
        x_new = np.array(x) - eta * der_x
        x     = list(x_new)
        values.append(f(x))
        
    return values