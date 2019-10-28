import numpy as np

from .util import construct_g2
from .util import construct_g1
from .util import construct_g
from .util import construct_f


from .der import construct_der_g2
from .der import construct_der_g1
from .der import construct_nabla_g
from .der import construct_nabla_f

tau   = np.exp(1)
L     = np.exp(1)
gamma = 1

g2    = construct_g2(L, gamma, tau)
g1    = construct_g1(L, gamma, tau)
g     = construct_g(g1, g2)
nu    = - g1(2*tau) + 4 * L * tau**2

f = construct_f(L, gamma, tau,nu, g, g1)   

der_g2    = construct_der_g2(L, gamma, tau)
der_g1    = construct_der_g1(L, gamma, tau)

nabla_g   = construct_nabla_g(g1, g2, der_g1, der_g2)
nabla_f   = construct_nabla_f(L, gamma, tau,nu, nabla_g, der_g1)   

def get_rand_vec(dims):
    x = np.random.standard_normal(dims)
    return x / np.linalg.norm(x)

t_thresh  = -1  
g_thresh  = np.exp(1) * gamma / 100
r         = np.exp(1) / 100
eta = 1 / (4 * L)

def experiment_pgd(x_0, iters):
    x = x_0
    d = len(x)
    t_noise   = - t_thresh -1 
    values = []

    for i in range(iters):
        der_x = nabla_f(x)
        der_x = np.array(der_x)
    
        if g_thresh >= np.linalg.norm(der_x) and  (i - t_noise > t_thresh):
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