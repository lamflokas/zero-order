import numpy as np

def construct_der_g2(L, gamma, tau):
    def fun(x):
        part_1       = 0
        part_2_up    = - 10 *(L+gamma) * 3* (x-2*tau)**2
        part_2_down  = tau**3
        part_2       = part_2_up / part_2_down
        part_3_up    = - 15 *(L+gamma) * 4 * (x-2*tau)**3
        part_3_down  = tau**4
        part_3       = part_3_up / part_3_down
        part_4_up    = - 6 *(L+gamma) * 5 * (x-2*tau)**4
        part_4_down  = tau**5
        part_4       = part_4_up / part_4_down
        return part_1 + part_2 + part_3 + part_4            
    return fun

def construct_der_g1(L, gamma, tau):
    
    # p
    def construct_poly(c_0, c_1, c_2, c_3, y_0):
        def poly(x):
            delta = x - y_0
            return c_3 * delta ** 3 \
                + c_2 * delta**2 \
                + c_1 * delta   \
                + c_0 
        return poly
    
    # y0, y1, f(y0), f(y1), f'(y0), f'(y1)
    def fun(y_0, y_1, a_0, a_1, a_2, a_3):
        S   = (a_1 - a_0) / (y_1 - y_0)
        c_0 = a_0    
        c_1 = a_2
        c_2 = (3* S - a_3 - 2* a_2) / (y_1 - y_0)
        c_3 = - (2* S - a_3 - a_2)/ ( (y_1 - y_0)**2 )
        return construct_poly(c_0, c_1, c_2, c_3, y_0)

    p = fun(y_0 = tau, 
            y_1 = 2 * tau,
            a_0 = - 2 * gamma * tau,
            a_1 = -4 * L * tau,
            a_2 = -2 * gamma,
            a_3 = 2 * L)
    
    return p

def construct_nabla_g(g1, g2, der_g1, der_g2):
    
    def nabla_g(x_1, x_2):
        return [der_g1(x_1) + x_2**2* der_g2(x_1) , 2 * x_2 * g2(x_1)]
    
    return nabla_g


def find_if_in_range(x, a, b):
    return all([ (y >=a and y <=b) for y in x])

def construct_nabla_f(L, gamma, tau,nu, nabla_g, der_g1):
    def f (x):  
        
        a_0   = [i for (i,y) in enumerate(x) if y >= 0]
        a_1   = [i for (i,y) in enumerate(x) if y < 0]
        x_abs = [abs(y) for y in x]
        
        for i in range(len(x)):
            if find_if_in_range(x_abs[i:], 0, tau) \
            and find_if_in_range(x_abs[:i], 2 * tau, 6 * tau) \
            and i< len(x)-1:
                
                compute_first_part = lambda x : 2 * L  * (x - 4 * tau) if x >=0 else 2 * L * (x + 4 * tau)
                first_part  = [compute_first_part(y) for y in x[:i]]
                
                second_part = - 2* gamma* x[i]
                
                thrid_part  = [L * 2 * y  for y in x[i+1:]]
                
                return first_part \
                        + [second_part] \
                        + thrid_part
            
            elif find_if_in_range(x_abs[:i], 2 * tau, 6 * tau)  \
            and find_if_in_range(x_abs[i+1:], 0, tau) \
            and find_if_in_range([x_abs[i]], tau, 2* tau) \
            and i< len(x)-1:
                
                compute_first_part = lambda x : 2 * L  * (x - 4 * tau) if x >=0 else 2 * L * (x + 4 * tau)
                first_part  = [compute_first_part(y) for y in x[:i]]
                
                second_part = nabla_g(x_abs[i], x[i+1])
                second_part[0] = second_part[0] if x[i] >= 0 else -second_part[0]
                
                thrid_part  = [L * 2 * y for y in x[i+2:]]
                return first_part \
                       + second_part \
                       + thrid_part
            
            elif find_if_in_range(x_abs[i:], 0, tau) \
            and  find_if_in_range(x_abs[:i], 2 * tau, 6 * tau) \
            and (i==len(x)-1):
                
                compute_first_part = lambda x : 2 * L  * (x - 4 * tau) if x >=0 else 2 * L * (x + 4 * tau)
                first_part  = [compute_first_part(y) for y in x[:i]]
                
                second_part = - 2* gamma* x[i]
                return first_part \
                        + [second_part]
            
            elif find_if_in_range(x_abs[:i], 2 * tau, 6 * tau) \
            and find_if_in_range([x_abs[i]], tau, 2* tau)\
            and (i==len(x)-1):
                
                compute_first_part = lambda x : 2 * L  * (x - 4 * tau) if x >=0 else 2 * L * (x + 4 * tau)
                first_part  = [compute_first_part(y) for y in x[:i]]
                
                second_part = der_g1(x_abs[i])
                second_part = second_part if x[i] >= 0 else -second_part
                
                return first_part \
                       + [second_part]
            
            elif find_if_in_range(x_abs, 2 * tau, 6 * tau) \
            and (i==len(x)-1):
                
                compute_first_part = lambda x : 2 * L  * (x - 4 * tau) if x >=0 else 2 * L * (x + 4 * tau)
                first_part  = [compute_first_part(y) for y in x]
                return first_part
   
        raise Exception
                
 
    return f