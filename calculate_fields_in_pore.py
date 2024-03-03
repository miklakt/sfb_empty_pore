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

def mobility_Rubinstein(phi, k, d, prefactor = 1):
    eps = np.where(phi==0, 0.0, 1/phi)
    m = eps * eps / (d * d)/prefactor
    m = m /(1.0 + m**k)**(1 / k)
    m = np.where(phi>0, m, 1.0)
    m = m
    return m

def mobility_Phillies(phi, beta, nu):
    m = np.where(phi==0, 1.0, np.exp(-beta*np.power(phi, nu)))
    return m

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

def add_walls(fields, exclude_volume = True, d = None):
    l1 = fields["l1"].squeeze()
    l2 = fields["l2"].squeeze()
    xlayers = fields["xlayers"].squeeze()
    ylayers = fields["ylayers"].squeeze()
    pore_radius = fields["r"].squeeze()
    wall_thickness = fields["s"].squeeze()

    W_arr = np.zeros((xlayers, ylayers))
    W_arr[l1:l1+wall_thickness+1, pore_radius:] = True
    if exclude_volume:
        if d is None:
            raise ValueError("No particle size specified")
        W_arr = ndimage.binary_dilation(W_arr, structure = generate_circle_kernel(d))
    fields["walls"] = W_arr

def calculate_pressure(fields, truncate = False):
    phi = fields["phi"].squeeze()
    chi = fields["chi_PS"].squeeze()
    fields["Pi"] = Pi(phi, chi, truncate)



@pickle_cache.pickle_lru_cache(purge_cache=True)
def calculate_fields(
        a0, a1, 
        chi_PC, chi, 
        wall_thickness, pore_radius, d, 
        exclude_volume = True, 
        truncate_pressure = False, 
        method = "convolve", 
        mobility_correction = "vol_average",
        mobility_model = "Rubinstein",
        **mobility_model_kwargs
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
        if mobility_correction != "none":
            if mobility_correction == "vol_average_corrected":
                average_corrected_phi = (a0 + a1*chi_PC)*phi
            elif mobility_correction == "vol_average":
                average_corrected_phi = phi
            fields["corrected_phi"] = convolve_particle_volume(average_corrected_phi.T, d, convolve_mode).T / volume(d)
        else:
            fields["corrected_phi"] = phi


        if mobility_model=="Rubinstein":
            if "prefactor" in mobility_model_kwargs:
                prefactor = mobility_model_kwargs["prefactor"]
            else:
                prefactor = 1
            k=1
            mobility = mobility_Rubinstein(fields["corrected_phi"], k, d, prefactor)
        
        elif mobility_model=="Phillies":
            beta = mobility_model_kwargs["beta"]
            nu = mobility_model_kwargs["nu"]
            mobility = mobility_Phillies(fields["corrected_phi"], beta, nu)

    mobility[fields["walls"]==True] = 0
    fields["mobility"] = mobility

    #==pore permeability
    #----pore with no polymer brush in it----
    #for the reference and normalization
    #empty_pore_conductivity = np.ones_like(fields["phi"])
    #empty_pore_conductivity[fields["walls"]==True] = 0

    #----permeability on z----
    #numerical integration over z-crossection in cylindrical coordinates
    r = np.arange(0, pore_radius)
    get_permeability_z = lambda x: np.sum(x[:, 0:pore_radius]*(2*r+1), axis = 1)#*2*np.pi
    fields["conductivity"]  = fields["mobility"]*np.exp(-fields["free_energy"])
    
    fields["permeability_z"] = get_permeability_z(fields["conductivity"])
    #fields["permeability_z_empty"] = get_permeability_z(empty_pore_conductivity)

    return fields

def integrate_permeability_over_z(fields, L):
    #====total permeability over region L====
    l1 = fields["l1"].squeeze()
    l2 = fields["l2"].squeeze()
    wall_thickness = fields["s"].squeeze()
    lb = l1-L
    rb = l1+wall_thickness+L+1 #+1 because of the offset after convolution
    get_permeability = lambda x: np.sum(x[lb:rb]**(-1))**(-1)
    #p = {}
    #p["permeability"] = get_permeability(fields["permeability_z"])
    #p["permeability_empty"] = get_permeability(fields["permeability_z_empty"])
    #return p
    return get_permeability(fields["permeability_z"])

#%%