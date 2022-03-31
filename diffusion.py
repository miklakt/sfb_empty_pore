from typing import Callable, Union, List
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

def D_eff(b: float, U_x : Union[Callable,List], half = False, parity = None):
    #interpolate if an array provided
    if isinstance(U_x, (list, tuple, np.ndarray)):
        print("Array")
        U_x = interp1d(range(len(U_x)), U_x)
    
    #position-independent diffusion coefficient
    if half:
        a = -b
    else:
        a = 0

    if parity=="even":
        def U_func(x):
            return U_x(abs(x))
    elif parity=="odd":
        def U_func(x):
            if x<0:
                return -U_x(abs(x))
            else:
                return U_x(x)
    else:
        U_func = U_x
    
    I = quad(lambda x: np.exp(U_func(x)), a, b)[0]
    D_eff_= (b-a)/I
    
    return D_eff_