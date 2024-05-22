# %%
import numpy as np
import pandas as pd
from scipy import ndimage
from functools import lru_cache
import utils
from particle_convolution import convolve_particle_surface, convolve_particle_volume
from joblib import Memory
import sfbox_utils
np.seterr(divide='ignore')

memory = Memory("__func_cache__", verbose=1)

import os
dirname = os.path.dirname(__file__)
master_empty = pd.read_pickle(os.path.join(dirname,"reference_table_empty_brush.pkl"))

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
    if prefactor==0:
        m = np.ones_like(phi)
        return m
    eps = np.where(phi==0, 0.0, 1/phi)
    m = eps * eps / (d * d)/prefactor
    m = m /(1.0 + m**k)**(1 / k)
    m = np.where(phi>0, m, 1.0)
    return m

def mobility_Phillies(phi, beta, nu):
    m = np.where(phi==0, 1.0, np.exp(-beta*np.power(phi, nu)))
    return m

def mobility_Hoyst(phi, d, N, alpha, delta, nu = 0.76):
    R_g = np.sqrt(N/6)
    phi_entangled = (9/(2*np.pi))/R_g
    xi = R_g*(phi/phi_entangled)**(-nu)
    # if d>R_g:
    #     b = R_g/xi
    # else:
    b = d/xi
    D0_D =  np.exp(alpha*b**delta)
    return 1/D0_D

# def mobility_Hoyst(phi, d, alpha, delta):
#     # R_g = np.sqrt(N/6)
#     # b = phi/R_g)
#     b = phi/d
#     D0_D = np.exp(alpha*b**delta)
#     return 1/D0_D

# def mobility_FoxFlory(phi, N):
#     eta = 1 + 0.425*np.sqrt(N)*phi
#     return 1/eta
    

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

def calculate_energy(fields, d, method, convolve_mode = "same"):
    #==free energy calculation==
    #----approximate method----
    if method == "approx":
        fields["surface"] = fields["gamma"]*surface(d)
        fields["osmotic"] = fields["Pi"]*volume(d)
    #----convolution method the output is staggered----
    if method == "convolve":
        fields["surface"] = convolve_particle_surface(fields["gamma"].T, d, convolve_mode).T
        fields["osmotic"] = convolve_particle_volume(fields["Pi"].T, d, convolve_mode).T
    if method == "no_free_energy":
        fields["surface"] = np.zeros_like(fields["Pi"])
        fields["osmotic"] = np.zeros_like(fields["Pi"])
    #----total free energy----
    fields["free_energy"] = fields["surface"] + fields["osmotic"]

def calculate_corrected_phi(fields, a0, a1, chi_PC, correction = "vol_average", d = None, convolve_mode = "valid"):
    phi = fields["phi"].squeeze()
    #print(d)
    if correction != "none":
        if correction == "chi_corrected":
            phi = (a0 + a1*chi_PC)*phi
            fields["corrected_phi"] = convolve_particle_volume(phi.T, d, convolve_mode).T / volume(d)
        elif correction == "vol_average":
            fields["corrected_phi"] = convolve_particle_volume(phi.T, d, convolve_mode).T / volume(d)
        else:
            raise ValueError("wrong correction mode")
    else:
        fields["corrected_phi"] = phi
    phi_err = 1e-10
    fields["corrected_phi"][(fields["corrected_phi"] > -phi_err)&(fields["corrected_phi"] < 0)] = 0.0

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

    elif model=="Hoyst":
        N = model_kwargs["N"]
        alpha = model_kwargs["alpha"]
        delta = model_kwargs["delta"]
        #mobility = mobility_Hoyst_old(fields[phi_arr], d, N, alpha, delta)
        mobility = mobility_Hoyst(fields[phi_arr], d, N, alpha, delta)
    
    elif model=="Fox-Flory":
        N = model_kwargs["N"]
        mobility = mobility_FoxFlory(fields[phi_arr], N)

    elif model == "none":
        xlayers = fields["xlayers"]
        ylayers = fields["ylayers"]
        mobility = np.ones((ylayers, xlayers))
    
    mobility[fields["walls"]==True] = 0
    fields["mobility"] = mobility

def calculate_conductivity(fields):
    fields["conductivity"]  = fields["mobility"]*np.exp(-fields["free_energy"])

def integrate_with_cylindrical_caps(
    field, 
    l1, wall_thickness, pore_radius, 
    xlayers, ylayers,
    first_cap_same_radius = False, #first cylindrical outside the pore has no offset,
    spheroid_correction = False
    ):

    if first_cap_same_radius:
        conductivity = integrate_with_cylindrical_caps(
            field, l1-1, wall_thickness+2,
            pore_radius, xlayers, ylayers,
            first_cap_same_radius=False
            )
        conductivity[l1-1] =conductivity[l1-1]+2*np.pi*pore_radius*field[l1-1, pore_radius]
        conductivity[l1+wall_thickness+1] =conductivity[l1+wall_thickness+1]+2*np.pi*pore_radius*field[l1+wall_thickness, pore_radius]
        return conductivity

    lb = 0
    rb = ylayers
    #r = np.arange(0, xlayers)
    conductivity = np.zeros(rb-lb)
    for i, z_ in enumerate(range(lb, rb)):
        #inside the pore
        if (z_>=l1) and (z_<=l1+wall_thickness):
            conductivity_ = np.pi*np.sum(field[z_, :pore_radius]*(2*np.arange(pore_radius)+1))
        else:
            if z_<l1:
                side = "left"
                el_a = z_
                el_b = l1
            elif z_>l1+wall_thickness:
                side = "right"
                el_a = l1+wall_thickness
                el_b = z_
            else:
                raise ValueError("z value is very strange")
            dist = el_b - el_a #how far we are form the edge
            cap_radius = pore_radius+dist
            if side == "left":
                z__ = z_
            if side == "right":
                z__ = z_-1
            
            if cap_radius>=xlayers:#outside simbox
                base_conductivity = np.pi*np.sum(field[z__, :xlayers]*(2*np.arange(xlayers)+1))
                base_conductivity = base_conductivity + np.pi*(cap_radius**2 - xlayers**2)
                element_conductivity = 2*np.pi*cap_radius*dist
            else:
                base_conductivity = np.pi*np.sum(field[z__, :cap_radius]*(2*np.arange(cap_radius)+1))
                element_conductivity = 2*np.pi*cap_radius*np.sum(field[el_a:el_b, cap_radius])

            conductivity_ = base_conductivity+element_conductivity
            
            if spheroid_correction:
                conductivity_ = conductivity_*(2*np.log(3))/np.pi
        conductivity[i] = conductivity_
    return conductivity

def integrate_conductivity_cylindrical_caps(
        fields, 
        #extend_to_infinity = True,#make correction for the boundary at inf
        spheroid_correction = True, #make correction for non-spheroid iso-potential surface 
        first_cap_same_radius = False #first cylindrical outside the pore has no offset
        ):
    l1 = fields["l1"]
    pore_radius = fields["r"]
    wall_thickness = fields["s"]
    xlayers = fields["xlayers"]
    ylayers = fields["ylayers"]
    permeability_z = integrate_with_cylindrical_caps(
        fields["conductivity"],
        l1, wall_thickness, pore_radius, xlayers, ylayers,
        spheroid_correction=spheroid_correction
        )
    if spheroid_correction:
        R_left = (-np.log(pore_radius/3 + l1) + np.log(pore_radius + l1))/(4*np.log(3)*pore_radius)
        R_right = (-np.log(pore_radius/3 + (ylayers-l1-wall_thickness)) + np.log(pore_radius + (ylayers-l1-wall_thickness)))/(4*np.log(3)*pore_radius)
    else:
        R_left = (-np.log(pore_radius/3 + l1) + np.log(pore_radius + l1))/(2*np.pi*pore_radius)
        R_right = (-np.log(pore_radius/3 + (ylayers-l1-wall_thickness)) + np.log(pore_radius + (ylayers-l1-wall_thickness)))/(2*np.pi*pore_radius)

    if not np.isclose(R_left, R_right):
        print("The simulation box seems to be asymmetric")

    permeability = np.sum(permeability_z**(-1))**(-1)
    fields["permeability_z"] = permeability_z
    #if extend_to_infinity:
    fields["permeability"] = (permeability**(-1) + R_left + R_right)**(-1)
    #else:
    #    fields["permeability"] = permeability
    fields["R_left"] = np.sum(permeability_z[:l1]**(-1))+R_left
    fields["R_right"] = np.sum(permeability_z[(l1+wall_thickness+1):]**(-1))+R_right
    fields["R_pore"] = np.sum(permeability_z[l1:(l1+wall_thickness+1)]**(-1)) \
        * wall_thickness/(wall_thickness+1) #correct that it was integrated one extra layer


def integrate_conductivity_Rayleigh(fields, L):
    #====total permeability over region L====
    l1 = fields["l1"]
    l2 = fields["l2"]
    pore_radius = fields["r"]
    wall_thickness = fields["s"]
    lb = l1-L
    rb = l1+wall_thickness+L+1 #+1 because of the offset after convolution
    
    r = np.arange(0, pore_radius)
    get_permeability_z = lambda x: np.sum(x[:, 0:pore_radius]*(2*r+1), axis = 1)*np.pi
    get_permeability_total = lambda x: np.sum(x[lb:rb]**(-1))**(-1)
    
    fields["permeability_z"] = get_permeability_z(fields["conductivity"])
    fields["permeability"] = get_permeability_total(fields["permeability_z"])


def calculate_partition_coefficient(fields, cutoff_phi = 1e-5):
    xlayers = fields["xlayers"]
    volume = np.ones_like(fields["phi"])
    volume[:] = 2*np.pi*(np.arange(fields["xlayers"])+1)
    fields["PC"] = np.sum((np.exp(-fields["free_energy"])*volume)[fields["phi"]>cutoff_phi])\
                                                   /np.sum(volume[fields["phi"]>cutoff_phi])

@memory.cache
def calculate_fields(
        a0, a1, 
        chi_PC, chi, 
        wall_thickness, pore_radius, d,
        sigma,
        exclude_volume = True, 
        truncate_pressure = False, 
        method = "convolve",
        convolve_mode = "valid",
        mobility_correction = "vol_average",
        mobility_model = "Rubinstein",
        mobility_model_kwargs = {},
        cutoff_phi = 1e-5,
        ):
    fields = utils.get_by_kwargs(master_empty, chi_PS = chi, s = wall_thickness, r = pore_radius, sigma = sigma).squeeze()
    fields["phi"] = fields.dataset["phi"].squeeze().T

    add_walls(fields, exclude_volume, d)
    calculate_pressure(fields, truncate_pressure)
    calculate_gamma(fields, a0, a1, chi_PC)

    #convolve_mode = "same"

    calculate_energy(fields, d, method, convolve_mode)
    calculate_corrected_phi(fields, a0=a0, a1=a1, chi_PC = chi_PC, correction=mobility_correction, d=d, convolve_mode=convolve_mode)
    calculate_mobility(fields, d, mobility_model, mobility_model_kwargs, phi_arr="corrected_phi")
    calculate_conductivity(fields)
    calculate_partition_coefficient(fields, cutoff_phi)

    return fields


def empty_pore_permeability(D, r, s):
    return 2*D*r/(1 + 2*s/(r*np.pi))


def calculate_permeability(
        a0, a1, pore_radius, wall_thickness,
        d, chi_PS, chi_PC,
        sigma, #=0.03
        exclude_volume,#=True 
        truncate_pressure,#=False 
        method,#= "convolve",
        convolve_mode,# = "same",
        mobility_correction,# = "vol_average",
        mobility_model,#, = "Rubinstein",
        mobility_model_kwargs,
        integration= "cylindrical_caps", #="Rayleigh"|"Caps",
        integration_kwargs = {},
        cutoff_phi = 1e-5,
        ):
    
    fields = calculate_fields(
        a0=a0, a1=a1, d=d, sigma = sigma,
        chi_PC=chi_PC, chi=chi_PS,
        wall_thickness=wall_thickness,
        pore_radius=pore_radius,
        exclude_volume=exclude_volume,
        mobility_correction=mobility_correction,
        mobility_model = mobility_model,
        truncate_pressure=truncate_pressure,
        method = method,
        convolve_mode = convolve_mode,
        mobility_model_kwargs = mobility_model_kwargs,
        cutoff_phi = cutoff_phi,
        )

    result = dict(
        a0=a0, a1=a1, d=d, sigma = sigma,
        chi_PC=chi_PC, chi=chi_PS,
        wall_thickness=wall_thickness,
        pore_radius=pore_radius,
        exclude_volume=exclude_volume,
        mobility_correction=mobility_correction,
        mobility_model = mobility_model,
        truncate_pressure=truncate_pressure,
        method = method,
        convolve_mode = convolve_mode,
        mobility_model_kwargs = mobility_model_kwargs,
        integration_kwargs = integration_kwargs,
        cutoff_phi = cutoff_phi,
    )
    
    if integration=="cylindrical_caps":
        integrate_conductivity_cylindrical_caps(fields, **integration_kwargs)#, extend_to_infinity=True)
    elif integration=="Rayleigh":
        integrate_conductivity_Rayleigh(fields, **integration_kwargs)
    else:
        raise ValueError("Wrong integration type")

    einstein_factor = 1/(3*np.pi*d)
    result["einstein_factor"] = einstein_factor

    result["thin_empty_pore"] = empty_pore_permeability(1, pore_radius-d/2, 0)*einstein_factor
    result["thick_empty_pore"] = empty_pore_permeability(1, pore_radius-d/2, wall_thickness)*einstein_factor
    result["permeability"] = fields["permeability"]*einstein_factor
    result["permeability_z"] = fields["permeability_z"]*einstein_factor
    result["PC"] = fields["PC"]

    if integration=="cylindrical_caps":
        result["R_left"] = fields["R_left"]/einstein_factor
        result["R_right"] = fields["R_right"]/einstein_factor
        result["R_pore"] = fields["R_pore"]/einstein_factor

    return result

#%%
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
    d=8
    parameters = dict(
        a0 = 0.70585835,
        a1 = -0.31406453,
        chi_PC = -1.0,
        chi = 0.5,
        wall_thickness=52,
        pore_radius=26,
        d = d,
        sigma = 0.02,
        exclude_volume = True, 
        truncate_pressure = False,
        #method = "no_free_energy",
        method = "convolve",
        convolve_mode = "same",
        mobility_correction = "vol_average",
        mobility_model = "Rubinstein",
        mobility_model_kwargs = dict(prefactor = 1),
        #mobility_model = "none",
        #mobility_model_kwargs = {},
        #mobility_model = "Hoyst",
        #mobility_model_kwargs = {'alpha': 1.63, 'delta': 0.89, 'N':300},
        #mobility_model_kwargs = dict(beta = 8, nu = 0.76)
    )

    fields = calculate_fields(**parameters)
    parameters["chi_PS"] = parameters["chi"]
    del parameters["chi"]
    perm = calculate_permeability(**parameters,         
                integration= "cylindrical_caps",
                integration_kwargs = dict(spheroid_correction = True)
                )

    fields["volume"] = np.ones_like(fields["phi"])
    fields["volume"][:] = 2*np.pi*(np.arange(fields["xlayers"])+1)
    #perm
# %%
    r_cut = 50
    z_cut = 30
    plot_heatmap(fields, r_cut, z_cut, keys = ["corrected_phi", "free_energy", "osmotic", "conductivity"])
# %%
    l1 = fields["l1"]
    wall_thickness = fields["s"]
    pore_radius = fields["r"]
    xlayers = fields["xlayers"]
    ylayers = fields["ylayers"]
    field = np.ones_like(fields["conductivity"])
    field = fields["conductivity"]
    conduct = integrate_with_cylindrical_caps(field, l1, wall_thickness, pore_radius, xlayers, ylayers, spheroid_correction=True)
    #conduct2 = integrate_with_cylindrical_caps(field, l1-2, wall_thickness+4, int(pore_radius), xlayers, ylayers, spheroid_correction=True)
    #conduct2[:l1-2] = conduct[:l1-2]
# %%
    x = np.arange(ylayers) - ylayers/2
    #plt.plot(conduct[:int(292/2-26)])
    plt.plot(x, conduct**-1)
    #plt.plot(x, conduct2**-1)
    plt.axhline(1/(np.pi*(pore_radius-d/2)**2), color = "red")
    plt.axvline(-wall_thickness/2, color = "black", linestyle = "--")
    plt.axvline(wall_thickness/2, color = "black", linestyle = "--")
    #plt.plot(field)
# %%
    einstein_factor = 1/(3*np.pi*d)
    R_thin = 1/empty_pore_permeability(1, pore_radius-d/2, 0)/einstein_factor/2
    R_pore = wall_thickness/(np.pi*(pore_radius-d/2)**2)/einstein_factor
    print(f"{R_thin=} {R_pore=}")
# %%
    np.sum(conduct[l1:l1+wall_thickness+1]**(-1))/einstein_factor
# %%
    
# %%
