from scipy import optimize
import torch
import numpy as np

################################### GC in Scipy #####################################
def f_scipy(x, *args):
    # expression a*u**2 + b*u*v + c*v**2 + d*u + e*v + f 
    u, v = x
    a, b, c, d, e, f = (2, 3, 7, 8, 9, 10)  # parameter values
    return a*u**2 + b*u*v + c*v**2 + d*u + e*v + f    

def gradf_scipy(x, *args):
    u, v = x
    a, b, c, d, e, f = args
    gu = 2*a*u + b*v + d     # u-component of the gradient
    gv = b*u + 2*c*v + e     # v-component of the gradient
    return np.asarray((gu, gv))

def gc_scipy():
    args = (2, 3, 7, 8, 9, 10)  # parameter values
    x0 = np.asarray((0, 0))  # Initial guess.
    res1 = optimize.fmin_cg(f_scipy, x0, fprime=gradf_scipy, args=args)

print('gc in scipy')        
gc_scipy()

######################################## GC in pytorch using Scipy ######################
a, b, c, d, e, f = torch.tensor([2, 3, 7, 8, 9, 10], dtype=torch.float32)

def f_torch(x):
    u, v = torch.tensor(x, dtype=torch.float32, requires_grad=False)
    r = a*u**2 + b*u*v + c*v**2 + d*u + e*v + f
    return r.numpy()

def gradf_torch(x):
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    def forward(x):
        u, v = x
        return a*u**2 + b*u*v + c*v**2 + d*u + e*v + f
    r = forward(x) # forward
    r.backward()   # backward
    return x.grad.numpy() # get gradient of x

def gc_torch():
    x0 = np.asarray((0, 0))  # Initial guess.
    res1 = optimize.fmin_cg(f_torch, x0, fprime=gradf_torch)

print('\ngc in pytorch')  
gc_torch()



##########################  results ##########################

'''
gc in scipy
Optimization terminated successfully.
         Current function value: 1.617021
         Iterations: 4
         Function evaluations: 8
         Gradient evaluations: 8

gc in pytorch
Optimization terminated successfully.
         Current function value: 1.617022
         Iterations: 4
         Function evaluations: 8
         Gradient evaluations: 8
'''
