#%%
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import itertools

import numpy as np
import utils
import pandas as pd
from scipy.signal import convolve
import sfbox_utils
import seaborn as sns
#%%
def cylinder_r0_kernel(radius:int, height:int = None):
    if height is None:
        height = radius*2
    r = np.arange(radius)
    volume_r = np.pi*(2*r+1)
    volume = np.tile(volume_r, (height,1)).T
    surface = np.zeros_like(volume)

    surface[:, 0] = surface[:,-1] = volume_r
    surface[-1,:] =surface[-1,:] + 2*np.pi*radius
    return volume, surface

def cylinder_volume_surface(radius:int, height:int = None):
    if height is None:
        height = radius*2
    V = np.pi*radius**2*height
    S = 2*np.pi*radius**2 + 2*np.pi*radius*height
    return V, S

def Pi(phi, chi_PS, trunc = False):
    Pi_=-np.log(1-phi) - phi - chi_PS*phi**2
    if trunc:
        Pi_[Pi_<1e-16]=0
    return Pi_

def gamma(chi_PS, chi_PC, phi, X):
    a0, a1 = X
    chi_crit = 6*np.log(5/6)
    phi_corrected = (a0 + a1*chi_PC)*phi
    chi_ads = chi_PC - chi_PS*(1-phi_corrected)
    #chi_ads = chi_PC - chi_PS*(1-phi)
    gamma = (chi_ads - chi_crit)*phi_corrected/6
    #gamma = (chi_ads - chi_crit)*phi/6
    return gamma

# def gamma2(chi_PS, chi_PC, phi, X):
#     a0, a1, a2 = X
#     chi_crit = 6*np.log(5/6)
#     #phi_corrected = (a0 + a1*chi_PC + a2*chi_PS)*phi
#     chi_ads = a0*chi_PC - a1*chi_PS*chi_PC + a2*chi_PS
#     gamma = (chi_ads - chi_crit)*phi_corrected/6
#     return gamma

def free_energy_cylinder(radius, data, chi_PS, chi_PC, gamma_func, X_args, trunc = False):
    volume, surface = cylinder_r0_kernel(radius)
    phi = data.dataset["phi"].squeeze()
    if np.shape(phi)[0] == 1:
        phi = np.tile(phi, (radius, 1))
    phi = np.pad(phi[0:radius], ((0, 0),(radius,radius-1)))
    Pi_arr = Pi(phi, chi_PS, trunc)
    gamma_arr = gamma_func(chi_PS, chi_PC, phi, X_args)
    osmotic = convolve(Pi_arr, volume, 'valid')[0]
    surface = convolve(gamma_arr, surface, 'valid')[0]
    #extra = X_args[2]*radius**2
    return osmotic, surface#, extra

def free_energy_approx(radius, data, chi_PS, chi_PC, gamma_func, X_args, trunc = False):
    volume, surface =cylinder_volume_surface(radius)
    phi = data.dataset["phi"].squeeze()[0, :]
    Pi_arr = Pi(phi, chi_PS, trunc)
    gamma_arr = gamma_func(chi_PS, chi_PC, phi, X_args)
    osmotic = Pi_arr*volume
    surface = gamma_arr*surface
    return osmotic, surface

def create_cost_function(df, df_empty, gamma_func):
    def cost_function(X):
        cost = np.array([])
        for (chi_PS, chi_PC, pw), group in df.groupby(by = ["chi_PS", "chi_PC", "pw"]):
            empty_pore_data = utils.get_by_kwargs(df_empty, chi_PS = chi_PS)
            if empty_pore_data.empty:
                continue
            osm, sur = free_energy_cylinder(
                int(pw/2), empty_pore_data, 
                chi_PS, chi_PC, gamma_func, X,
                )
            tot = osm+sur
            delta_fe = group.apply(lambda _: _.free_energy - tot[int(_.pc+len(tot)//2)], axis = 1)
            delta_fe = np.cbrt(delta_fe)
            cost = np.concatenate([cost, delta_fe.to_numpy()])
        return cost
    return cost_function