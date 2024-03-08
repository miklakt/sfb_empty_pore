# %%
import numpy as np
import pandas as pd
from scipy import ndimage
from functools import lru_cache
import utils
from particle_convolution import convolve_particle_surface, convolve_particle_volume
#import pickle_cache
from joblib import Memory
import sfbox_utils
np.seterr(divide='ignore')

memory = Memory("__func_cache__", verbose=1)
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

def mobility_Holyst(phi, alpha, delta, N):
    R_g = np.sqrt()

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
    l1 = fields["l1"]
    l2 = fields["l2"]
    xlayers = fields["xlayers"]
    ylayers = fields["ylayers"]
    pore_radius = fields["r"]
    wall_thickness = fields["s"]

    W_arr = np.zeros((ylayers,xlayers))
    W_arr[l1:l1+wall_thickness+1, pore_radius:] = True
    if exclude_volume:
        if d is None:
            raise ValueError("No particle size specified")
        W_arr = ndimage.binary_dilation(W_arr, structure = generate_circle_kernel(d))
    fields["walls"] = W_arr

def calculate_pressure(fields, truncate = False):
    phi = fields["phi"]
    chi = fields["chi_PS"]
    fields["Pi"] = Pi(phi, chi, truncate)

def calculate_gamma(fields, a0, a1, chi_PC):
    phi = fields["phi"]
    chi = fields["chi_PS"]
    #chi_PC = fields["chi_PC"]
    fields["gamma"] = gamma(chi, chi_PC, phi, a0, a1)
    fields["a0"] = a0
    fields["a1"] = a1

def calculate_energy(fields, d, method):
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

def calculate_corrected_phi(fields, a0, a1, chi_PC, correction = "average", d = None, convolve_mode = "valid"):
    phi = fields["phi"]
    #chi_PC = fields["chi_PC"]
    if correction != "none":
        if correction == "vol_average_corrected":
            average_corrected_phi = (a0 + a1*chi_PC)*phi
        elif correction == "average":
            average_corrected_phi = phi
            fields["corrected_phi"] = convolve_particle_volume(average_corrected_phi.T, d, convolve_mode).T / volume(d)
    else:
        fields["corrected_phi"] = phi

def calculate_mobility(fields, d, model, model_kwargs = {}, phi_arr = "phi"):
    if model=="Rubinstein":
        if "prefactor" in model_kwargs:
            prefactor = model_kwargs["prefactor"]
        else:
            prefactor = 1
        k=1
        mobility = mobility_Rubinstein(fields[phi_arr], k, d, prefactor)
    
    elif model=="Phillies":
        beta = model_kwargs["beta"]
        nu = model_kwargs["nu"]
        mobility = mobility_Phillies(fields[phi_arr], beta, nu)

    elif model == "none":
        xlayers = fields["xlayers"]
        ylayers = fields["ylayers"]
        mobility = np.ones((xlayers, ylayers))
    
    mobility[fields["walls"]==True] = 0
    fields["mobility"] = mobility

def calculate_conductivity(fields):
    pore_radius = fields["r"]
    r = np.arange(0, pore_radius)
    
    get_permeability_z = lambda x: np.sum(x[:, 0:pore_radius]*(2*r+1), axis = 1)#*2*np.pi
    fields["conductivity"]  = fields["mobility"]*np.exp(-fields["free_energy"])
    
    fields["permeability_z"] = get_permeability_z(fields["conductivity"])

def calculate_permeability(fields, L):
    #====total permeability over region L====
    l1 = fields["l1"].squeeze()
    l2 = fields["l2"].squeeze()
    wall_thickness = fields["s"].squeeze()
    lb = l1-L
    rb = l1+wall_thickness+L+1 #+1 because of the offset after convolution
    get_permeability = lambda x: np.sum(x[lb:rb]**(-1))**(-1)
    return get_permeability(fields["permeability_z"])


#@pickle_cache.pickle_lru_cache(purge_cache=True)
@memory.cache
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
    fields["phi"] = fields.dataset["phi"].squeeze().T

    add_walls(fields, exclude_volume, d)
    calculate_pressure(fields, truncate_pressure)
    calculate_gamma(fields, a0, a1, chi_PC)

    if method == "convolve":
        convolve_mode = "valid"
    if method == "convolve_same":
        convolve_mode = "same"

    calculate_energy(fields, d, method)
    calculate_corrected_phi(fields, a0, a1, mobility_correction, d, convolve_mode)
    calculate_mobility(fields, d, mobility_model, mobility_model_kwargs)
    calculate_conductivity(fields)
    return fields

def integrate_permeability_over_z(fields, L):
    l1 = fields["l1"].squeeze()
    l2 = fields["l2"].squeeze()
    wall_thickness = fields["s"].squeeze()
    lb = l1-L
    rb = l1+wall_thickness+L+1 #+1 because of the offset after convolution
    get_permeability = lambda x: np.sum(x[lb:rb]**(-1))**(-1)
    return get_permeability(fields["permeability_z"])

def plot_heatmap(fields, r_cut, z_cut, keys, **kwargs):
    from heatmap_explorer import plot_heatmap_and_profiles
    wall_thickness = fields["s"]
    l1 = fields["l1"]
    def cut_and_mirror(arr):
        cut = arr.T[0:r_cut, l1-z_cut:l1+wall_thickness+z_cut]
        return np.vstack((np.flip(cut), cut[:,::-1]))
    extent = [-z_cut-wall_thickness/2, z_cut+wall_thickness/2, -r_cut, r_cut]
    for key in keys:
        mask = cut_and_mirror(fields["walls"])
        fig = plot_heatmap_and_profiles(
            cut_and_mirror(fields[key]).T,
            y0=-r_cut,
            x0=-z_cut-wall_thickness/2,
            ylabel="$r$",
            xlabel = "$z$",
            zlabel=key,
            update_zlim=False,
            hline_y=int(z_cut+wall_thickness/2),
            vline_x=r_cut,
            mask = mask.T,
            **kwargs
            )
        fig.show()
    return fig

#%%
if __name__ == "__main__":
    parameters = dict(
        a0 = 0.70585835,
        a1 = -0.31406453, 
        chi_PC = -1, 
        chi = 0.5, 
        wall_thickness=52, 
        pore_radius=26, 
        d = 8, 
        exclude_volume = True, 
        truncate_pressure = False, 
        method = "convolve", 
        mobility_correction = "vol_average",
        mobility_model = "Rubinstein",
    )

    field = calculate_fields(**parameters)
# %%
r_cut = 50
z_cut = 30
plot_heatmap(field, r_cut, z_cut, keys = ["phi", "Pi", "gamma", "free_energy", "mobility", "conductivity"])

