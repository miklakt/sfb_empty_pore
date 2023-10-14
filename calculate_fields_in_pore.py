# %%
import numpy as np
import pandas as pd
from scipy import ndimage
from functools import lru_cache
import utils
from particle_convolution import convolve_particle_surface, convolve_particle_volume
import pickle_cache
np.seterr(divide='ignore')

master_empty = pd.read_pickle("reference_table_empty_brush.pkl")

def volume(d):
    return np.pi*d**3/6

def surface(d):
    return np.pi*d**2

def gamma(chi_PS, chi_PC, phi, a0, a1):
    #a0, a1 = X
    chi_crit = 6*np.log(5/6)
    phi_corrected = (a0 + a1*chi_PC)*phi
    chi_ads = chi_PC - chi_PS*(1-phi_corrected)
    gamma = (chi_ads - chi_crit)*phi_corrected/6
    return gamma

def Pi(phi, chi_PS, trunc = True):
    Pi_=-np.log(1-phi) - phi - chi_PS*phi**2
    if trunc:
        Pi_[Pi_<1e-16]=0
    return Pi_

def surface_free_energy(phi, chi_PS, chi_PC, d, a0, a1):
    return surface(d)*gamma(chi_PS, chi_PC, phi, a0, a1)

def volume_free_energy(phi, chi_PS, d, trunc=True):
    return Pi(phi, chi_PS, trunc)*volume(d)

def free_energy_phi(phi, chi_PS, chi_PC, d, a0, a1):
    return surface_free_energy(phi, chi_PS, chi_PC, d, a0, a1)+volume_free_energy(phi, chi_PS, d)

def mobility_phi(phi, k, d):
    eps = np.where(phi==0, 0.0, 1/phi)
    m = eps * eps / (d * d)
    m = m /(1.0 + m**k)**(1 / k)
    m = np.where(phi>0, m, 1.0)
    return m

#def integrate_cylinder(array_zr):
#    Nz, Nr = np.shape(array_zr)
#    H = Nz
#    A = Nr**2/2
#    r = np.arange(0, Nr)
#    element_volume = 2*r
#    element_volume[0] = 1/4
#    P_z = np.sum(array_zr*element_volume, axis = 1)**(-1)
#    P = np.sum(P_z)**(-1)
#    return P

@lru_cache()
def generate_circle_kernel(d):
    radius = d/2
    a = np.zeros((d, d), dtype =bool)
    radius2 = radius**2
    for i in range(d):
        for j in range(d):
            distance2 = (radius-i-0.5)**2 + (radius-j-0.5)**2
            if distance2<radius2:
                a[i,j] = True
    return a

@pickle_cache.pickle_lru_cache(purge_cache=False)
def calculate_fields(
        a0, a1, 
        chi_PC, chi, 
        wall_thickness, pore_radius, d, 
        exclude_volume = True, 
        truncate_pressure = False, 
        method = "convolve", 
        mobility_correction = "vol_average"
        ):
    fields = utils.get_by_kwargs(master_empty, chi_PS = chi, s = wall_thickness, r = pore_radius).squeeze()
    phi = fields.dataset["phi"].squeeze().T
    fields["phi"] = phi
    l1 = fields["l1"].squeeze()
    l2 = fields["l2"].squeeze()
    
    #==create walls==
    W_arr = np.zeros_like(phi)
    W_arr[l1:l1+wall_thickness+1, pore_radius:] = True
    #----excluded volume----
    if exclude_volume:
        W_arr = ndimage.binary_dilation(W_arr, structure = generate_circle_kernel(d))
    fields["walls"] = W_arr

    #==calculate osmotic pressure and surface coef==
    fields["Pi"] = Pi(phi, chi, truncate_pressure)
    fields["gamma"] = gamma(chi, chi_PC, phi, a0, a1)

    #==free energy calculation==
    #----approximate method----
    if method == "approx":
        fields["surface"] = fields["gamma"]*surface(d)
        fields["osmotic"] = fields["Pi"]*volume(d)
    #----convolution method the output is staggered----
    if method == "convolve":
        convolve_mode = "valid"
        fields["surface"] = convolve_particle_surface(fields["gamma"].T, d, convolve_mode).T
        fields["osmotic"] = convolve_particle_volume(fields["Pi"].T, d, convolve_mode).T
    #----convolution method the output is centered----
    if method == "convolve_same":
        convolve_mode = "same"
        fields["surface"] = convolve_particle_surface(fields["gamma"].T, d, convolve_mode).T
        fields["osmotic"] = convolve_particle_volume(fields["Pi"].T, d, convolve_mode).T
    #----total free energy----
    fields["free_energy"] = fields["surface"] + fields["osmotic"]

    #==conductivity==
    if mobility_correction == "no_mobility":
        mobility = np.ones_like(fields["phi"])
        mobility[fields["walls"]==True] = 0
        fields["mobility"] = mobility
    else:
        if mobility_correction == "vol_average_corrected":
            average_corrected_phi = (a0 + a1*chi_PC)*phi
        elif mobility_correction == "vol_average":
            average_corrected_phi = phi
        fields["average_corrected_phi"] = convolve_particle_volume(average_corrected_phi.T, d, convolve_mode).T / volume(d)
    
        mobility = mobility_phi(fields["average_corrected_phi"], 1, d)
        mobility[fields["walls"]==True] = 0
        fields["mobility"] = mobility

    fields["conductivity"]  = fields["mobility"]*np.exp(-fields["free_energy"])

    #==pore permeability
    #----pore with no polymer brush in it----
    #for the reference and normalization
    empty_pore_conductivity = np.ones_like(fields["phi"])
    empty_pore_conductivity[fields["walls"]==True] = 0

    #----permeability on z----
    #numerical integration over z-crossection in cylindrical coordinates
    r = np.arange(0, pore_radius)
    #a_z = 2*r
    #a_z[0] = 1/4
    get_permeability_z = lambda x: np.sum(x[:, 0:pore_radius]*(2*r+1), axis = 1)#*2*np.pi
    
    fields["permeability_z"] = get_permeability_z(fields["conductivity"])
    fields["permeability_z_empty"] = get_permeability_z(empty_pore_conductivity)

    return fields

def integrate_permeability_over_z(fields, L):
    #====total permeability over region L====
    l1 = fields["l1"].squeeze()
    l2 = fields["l2"].squeeze()
    wall_thickness = fields["s"].squeeze()
    lb = l1-L
    rb = l1+wall_thickness+L+1 #+1 because of the offset after convolution
    get_permeability = lambda x: np.sum(x[lb:rb]**(-1))**(-1)
    p = {}
    p["permeability"] = get_permeability(fields["permeability_z"])
    p["permeability_empty"] = get_permeability(fields["permeability_z_empty"])
    return p

#%%