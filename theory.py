#%%
from typing import Callable, Union, List
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import functools
#import ascf_pb
import scipy.stats as st
from scipy import signal

def D_eff(U_x : Union[Callable,List], b: float, half = False, parity = None):
    #interpolate if an array provided
    if isinstance(U_x, (list, tuple, np.ndarray)):
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

def volume(w, h = None):
    if h is None:
        h=w
    return np.pi*w**2/4*h

def surface(w, h = None):
    if h is None:
        h=w
    return np.pi*w*h+np.pi*w**2/2

def gamma(a1, a2, chi_PS, chi_PC, phi):
    chi_crit = 6*np.log(5/6)
    chi_ads = chi_PC - chi_PS*(1-phi)
    gamma = (chi_ads - chi_crit)*(a1*phi+a2*phi**2)
    return gamma

def Pi(phi, chi_PS):
    return -np.log(1-phi) - phi - chi_PS*phi**2

def surface_free_energy(phi, a1, a2, chi_PS, chi_PC, w, h=None):
    return surface(w, h)*gamma(a1, a2, chi_PS, chi_PC, phi)

def volume_free_energy(phi, chi_PS, w, h=None):
    return Pi(phi, chi_PS)*volume(w,h)

@functools.lru_cache()
def free_energy_phi(phi, a1, a2, chi_PS, chi_PC, w, h=None):
    return surface_free_energy(phi, a1, a2, chi_PS, chi_PC, w, h)+volume_free_energy(phi, chi_PS, w, h)


def phi_corrected_1d(eps, pos, **kwargs):
    phi_func = np.vectorize(ascf_pb.phi(**kwargs))
    m = eps*3
    x = np.arange(-m, m, 1)
    gaussian = np.exp(-(x/eps)**2/2)
    x = np.arange(int(pos - m), int(pos + m))
    phi = phi_func(z=x)
    kernel =  gaussian/np.sum(gaussian)
    return np.sum(phi*kernel)

def cyl_weights(m, mirror = False, norm = False):
    if m%2 != 0:
        raise ValueError("Odd values are not implemented")
    else:
        kernel = np.tile(np.arange(1,m,2), (int(m),1)).T*np.pi
        if mirror:
            kernel = np.vstack([np.flip(kernel, axis = 0),kernel])
        if norm:
            kernel = kernel/np.sum(kernel)
    return kernel

def gauss_kernel(m, sigma, mirror = False, norm = True):
    """Returns a 2D Gaussian kernel.""" 
    x = np.arange(0.5,m/2+1.5)
    cdf = st.norm.cdf(x, scale = sigma)
    kern1d = np.diff(cdf)
    kernel = np.outer(kern1d, kern1d)
    kernel = np.hstack([np.flip(kernel, axis = 1),kernel])
    if mirror:
        kernel = np.vstack([np.flip(kernel, axis = 0),kernel])
    kernel = kernel/np.sum(kernel)
    return kernel

def gauss_kernel_new(m, sigma, mirror = False, norm = True):
    window = signal.windows.gaussian(M = m, std=sigma)
    kernel = np.outer(window, window)
    if not mirror:
        kernel = np.split(kernel, 2)[0]
    if norm:
        kernel = kernel/np.sum(kernel)
    return kernel

def gauss_kernel_cyl(m, sigma, mirror = False):
    kernel = gauss_kernel(m,sigma,mirror, norm = False)*cyl_weights(m,mirror,norm=False)
    kernel = kernel/np.sum(kernel)
    return kernel

def correct_phi(phi_2d : np.ndarray, kernel : np.ndarray):
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(phi_2d, kernel.shape)[0]
    phi_corrected = [np.tensordot(window, kernel) for window in windows]
    phi_corrected = np.pad(phi_corrected, (kernel.shape[0]-1,kernel.shape[0]))
    return phi_corrected
# %%
if __name__ == "__main__":
    def box_function(A, width, pad):
        def box(z):
            if (z<pad) or (z>pad+width):
                return 0
            else:
                return A
        return box

    import matplotlib.pyplot as plt
    A = -np.linspace(0,2)
    pad =-A
    width = 1
    D = [D_eff(box_function(A_,width, pad_), b = width+2*pad_) for A_, pad_ in  zip(A,pad)]
    plt.plot(A, D)
# %%
