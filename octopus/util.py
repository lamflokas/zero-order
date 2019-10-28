import numpy as np

def construct_g2(L, gamma, tau):
    def fun(x):
        part_1 = -gamma
        part_2_up    = - 10 *(L+gamma) * (x-2*tau)**3
        part_2_down  = tau**3
        part_2       = part_2_up / part_2_down
        part_3_up    = - 15 *(L+gamma) * (x-2*tau)**4
        part_3_down  = tau**4
        part_3       = part_3_up / part_3_down
        part_4_up    = - 6 *(L+gamma) * (x-2*tau)**5
        part_4_down  = tau**5
        part_4       = part_4_up / part_4_down
        return part_1 + part_2 + part_3 + part_4            
    return fun

def construct_g1(L, gamma, tau):
    # Anti-derivative of p
    def construct_poly(c_0, c_1, c_2, c_3, y_0):
        def poly(x):
            delta = x - y_0
            return c_3 * delta ** 4 / 4 \
                + c_2 * delta**3 / 3 \
                + c_1 * delta**2 /2  \
                + c_0 * delta
        return poly
    
    # y0, y1, f(y0), f(y1), f'(y0), f'(y1)
    def fun(y_0, y_1, a_0, a_1, a_2, a_3):
        S   = (a_1 - a_0) / (y_1 - y_0)
        c_0 = a_0    
        c_1 = a_2
        c_2 = (3* S - a_3 - 2* a_2) / (y_1 - y_0)
        c_3 = - (2* S - a_3 - a_2)/ ( (y_1 - y_0)**2 )
        return construct_poly(c_0, c_1, c_2, c_3, y_0)

    anti_p = fun(y_0 = tau, 
                 y_1 = 2 * tau,
                 a_0 = - 2 * gamma * tau,
                 a_1 = -4 * L * tau,
                 a_2 = -2 * gamma,
                 a_3 = 2 * L)
    
    return lambda x : anti_p(x) - anti_p(tau) - gamma * tau**2

def construct_g(g1, g2):
    def g(x_1, x_2):
        part_1 = g1(x_1)
        part_2 = x_2**2 * g2(x_1)
        return part_1 + part_2
    
    return g


def find_if_in_range(x, a, b):
    return all([ (y >=a and y <=b) for y in x])

def construct_f(L, gamma, tau,nu, g, g1):
    def f (x):
        
        a_0   = [i for (i,y) in enumerate(x) if y >= 0]
        a_1   = [i for (i,y) in enumerate(x) if y < 0]
        x_abs = [abs(y) for y in x]
        
        for i in range(len(x)):
            if find_if_in_range(x_abs[i:], 0, tau) \
            and find_if_in_range(x_abs[:i], 2 * tau, 6 * tau) \
            and i< len(x)-1:
                
                first_part  = [L * (y - 4 * tau) ** 2 for y in x_abs[:i]]
                second_part = - gamma*  (x[i] ** 2)
                third_part  = [L * y ** 2 for y in x[i+1:]]
                last_part   = -i * nu
                return sum(first_part) \
                        + second_part \
                        + sum(third_part) \
                        + last_part
            
            elif find_if_in_range(x_abs[:i], 2 * tau, 6 * tau)  \
            and find_if_in_range(x_abs[i+1:], 0, tau) \
            and find_if_in_range([x_abs[i]], tau, 2* tau) \
            and i< len(x)-1:
                
                first_part  = [L * (y - 4 * tau) ** 2 for y in x_abs[:i]]
                second_part = g(x_abs[i], x[i+1])
                third_part  = [L * y ** 2 for y in x[i+2:]]
                last_part   = -i * nu
                return sum(first_part) \
                        + second_part \
                        + sum(third_part) \
                        + last_part
            
            elif find_if_in_range(x_abs[i:], 0, tau) \
            and  find_if_in_range(x_abs[:i], 2 * tau, 6 * tau) \
            and (i==len(x)-1):
                
                first_part  = [L * (y - 4 * tau) ** 2 for y in x_abs[:i]]
                second_part = - gamma*  (x[i] ** 2)
                last_part   = -i * nu
                return sum(first_part) \
                        + second_part \
                        + last_part
            
            elif find_if_in_range(x_abs[:i], 2 * tau, 6 * tau) \
            and find_if_in_range([x_abs[i]], tau, 2* tau)\
            and (i==len(x)-1):
                
                first_part  = [L * (y - 4 * tau) ** 2 for y in x_abs[:i]]
                second_part = g1(x_abs[i])
                last_part   = -i * nu
                return sum(first_part) \
                        + second_part \
                        + last_part
            
            elif find_if_in_range(x_abs, 2 * tau, 6 * tau) \
            and (i==len(x)-1):
                
                first_part  = [L * (y - 4 * tau) ** 2 for y in x_abs]
                last_part   = -(i+1) * nu
                return sum(first_part) \
                        + last_part
   
        raise Exception
                
 
    return f