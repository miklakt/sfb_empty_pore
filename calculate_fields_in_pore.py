# %%
import numbers
import numpy as np
import pandas as pd
from scipy import ndimage
from functools import lru_cache
import utils
from particle_convolution import convolve_particle_surface, convolve_particle_volume
from joblib import Memory
np.seterr(divide='ignore')

memory = Memory("__func_cache__", verbose=1)

import os
dirname = os.path.dirname(__file__)
master_empty = pd.read_pickle(os.path.join(dirname,"pkl/reference_table_empty_brush.pkl"))

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
    m = eps * eps / (d * d)/prefactor**2
    m = m /(1.0 + m**k)**(1 / k)
    m = np.where(phi>0, m, 1.0)
    return m

def mobility_Phillies(phi, beta, nu):
    m = np.where(phi==0, 1.0, np.exp(-beta*np.power(phi, nu)))
    return m

def Haberman_correction_approximant(d, pore_radius):
    #wall drag correction
    if d/2>0.95*pore_radius:
        print("Particle is to big for Haberman correction")
        return None
    Pade_approximant = lambda x: (-2.6211*x**3 + 1.0626*x**2 + 1.9006*x**1 + 0.0089111)/(1-x)
    x_ = d/2 / pore_radius
    return np.exp(Pade_approximant(x_))

    
#%%
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

def subst_by_gel(fields, phi:float):
    l1 = fields["l1"]
    l2 = fields["l2"]
    xlayers = fields["xlayers"]
    ylayers = fields["ylayers"]
    pore_radius = fields["r"]
    wall_thickness = fields["s"]

    phi_arr = np.zeros((ylayers,xlayers))
    phi_arr[l1:l1+wall_thickness, :pore_radius] = phi
    fields["phi"] = phi_arr

def calculate_pressure(fields, truncate = False):
    phi = fields["phi"]
    chi = fields["chi_PS"]
    fields["Pi"] = Pi(phi, chi, truncate)

def calculate_gamma(fields, a0, a1, chi_PC):
    phi = fields["phi"]
    chi = fields["chi_PS"]
    fields["gamma"] = gamma(chi, chi_PC, phi, a0, a1)
    fields["a0"] = a0
    fields["a1"] = a1

def calculate_energy(fields, d, method):
    #==free energy calculation==
    #----approximate method----
    if method == "approx":
        #print("approximate method")
        fields["surface"] = fields["gamma"]*surface(d)
        fields["osmotic"] = fields["Pi"]*volume(d)
    #----convolution method the output is staggered----
    elif method == "convolve":
        #print("convolution method")
        #convolve_mode = "roll"#|trim
        fields["surface"] = convolve_particle_surface(fields["gamma"].T, d).T
        fields["osmotic"] = convolve_particle_volume(fields["Pi"].T, d).T
    elif method == "no_free_energy":
        fields["surface"] = np.zeros_like(fields["Pi"])
        fields["osmotic"] = np.zeros_like(fields["Pi"])
    else:
        raise ValueError("Wrong method in calculate free energy call")
    #----total free energy----
    fields["free_energy"] = fields["surface"] + fields["osmotic"]

def calculate_corrected_phi(fields, a0, a1, chi_PC, correction = "vol_average", d = None):
    phi = fields["phi"].squeeze()
    if correction != "none":
        #convolve_mode = "roll"#|trim
        if correction == "chi_corrected":
            phi = (a0 + a1*chi_PC)*phi
            fields["corrected_phi"] = convolve_particle_volume(phi.T, d).T / volume(d)
        elif correction == "vol_average":
            fields["corrected_phi"] = convolve_particle_volume(phi.T, d).T / volume(d)
        else:
            raise ValueError("wrong correction mode")
    else:
        fields["corrected_phi"] = phi
    phi_err = 1e-10
    fields["corrected_phi"][(fields["corrected_phi"] > -phi_err)&(fields["corrected_phi"] < 0)] = 0.0

def calculate_mobility(
        fields, d,
        model, model_kwargs = {}, 
        phi_arr = "phi", 
        Haberman_correction = False,
        stickiness = False,
        stickiness_model_kwargs = {},
        ):
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
        mobility = np.ones((ylayers, xlayers))

    else:
        raise ValueError("No diffusion model provided")

    if Haberman_correction:
        l1 = fields["l1"]
        wall_thickness = fields["s"]
        pore_radius = fields["r"]
        wall_drag_correction = Haberman_correction_approximant(d, pore_radius)
        mobility[l1:l1+wall_thickness, 0:pore_radius] = mobility[l1:l1+wall_thickness, 0:pore_radius]/wall_drag_correction
    
    mobility[fields["walls"]==True] = 0

    if stickiness:
        try:
            n_sites = stickiness_model_kwargs.pop("n_sites")
        except KeyError:
            n_sites = 1
        binding_enery = fields["surface"]/n_sites
        mobility = np.where(fields["surface"]<0,mobility*np.exp(binding_enery/2),mobility)
    
    fields["mobility"] = mobility


def calculate_conductivity(fields, D_0 = "Einstein"):
    d = fields["d"]
    if D_0 == "Einstein":
        D_0 = 1/(3*np.pi*d)
    elif D_0 is None:
        D_0 = 1
    elif isinstance(D_0, float|int):
        pass
    else:
        raise ValueError("Incorrect D_0")
    fields["D_0"] = D_0 
    fields["conductivity"]  = fields["mobility"]*np.exp(-fields["free_energy"])*D_0
    
def integrate_conductivity(fields, conductivity_key = "conductivity"):
    l1 = fields["l1"]
    pore_radius = fields["r"]
    wall_thickness = fields["s"]
    xlayers = fields["xlayers"]
    ylayers = fields["ylayers"]
    d = fields["d"]
    D_0 = fields["D_0"]
    conductivity = fields[conductivity_key]

    R_ext = 0
    if d>=2:
        z_ext = int(l1-d//2)
        pore_radius_apparent = int(pore_radius - d//2)
    else:
        z_ext = int(l1)
        pore_radius_apparent = pore_radius
    # z_ext = int(l1)
    # pore_radius_apparent = pore_radius
    for z in range(z_ext):
        dist = int(z_ext-z)
        cap_radius = int(dist+pore_radius_apparent)
        if cap_radius>=xlayers:#outside simbox
            base_conductivity = np.pi*np.sum(conductivity[z, :xlayers]*(2*np.arange(xlayers)+1))
            base_conductivity = base_conductivity + np.pi*(cap_radius**2 - xlayers**2)
            element_conductivity = 2*np.pi*D_0*dist
            # conductivity of the hollow tube with potential gradient applied to the inner and outer surface
            element_conductivity = element_conductivity/np.log(cap_radius/(cap_radius-1)) 
        else:
            base_conductivity = np.pi*np.sum(conductivity[z, :cap_radius]*(2*np.arange(cap_radius)+1))
            #cap_radius-1 is offset for elements being below the contour
            #no need too to offset for z, as the elements are to the right
            element_conductivity = 2*np.pi*np.sum(conductivity[z:z_ext, cap_radius])#*cap_radius
            # conductivity of the hollow tube with potential gradient applied to the inner and outer surface
            element_conductivity = element_conductivity/np.log(cap_radius/(cap_radius-1)) 

        conductivity_z = base_conductivity + element_conductivity
        #spheroid correction

        f = pore_radius_apparent/((dist*np.log((pore_radius_apparent + dist)/(pore_radius_apparent + dist - 1)) + (pore_radius_apparent + dist)**2)*(np.arctan(dist/pore_radius_apparent) - np.arctan((dist - 1)/pore_radius_apparent)))
        conductivity_z = conductivity_z*f
        R_ext = R_ext + conductivity_z**-1

    R_left = np.arctan(pore_radius_apparent/z_ext)/(2*np.pi*pore_radius_apparent*D_0)
    R_ext = R_ext+R_left
    R_ext = R_ext*2.0

    R_int = 0
    for z in range(z_ext, ylayers//2):
        conductivity_z = np.pi*np.sum(conductivity[z, :pore_radius]*(2*np.arange(pore_radius)+1))
        R_int = R_int + conductivity_z**-1
    R_int = R_int*2.0

    fields["R"] = R_int+R_ext
    fields["R_int"] = R_int
    fields["R_ext"] = R_ext


def calculate_partition_coefficient(fields, cutoff_phi = 1e-5):
    xlayers = fields["xlayers"]
    volume = np.ones_like(fields["phi"])
    volume[:] = 2*np.pi*(np.arange(fields["xlayers"])+1)
    fields["PC"] = np.sum((np.exp(-fields["free_energy"])*volume)[fields["phi"]>cutoff_phi])\
                                                   /np.sum(volume[fields["phi"]>cutoff_phi])
def empty_pore_permeability(D, r, s):
    return 2*D*r/(1 + 2*s/(r*np.pi))

def empty_pore_permeability_corrected(D, r, s, d):
    K = Haberman_correction_approximant(d, r)
    return 2*D*(r-d/2)/(1 + 2*K*(s+d)/((r-d/2)*np.pi))

def pad_phi_field(fields, pad_sides, pad_top):
    fields["ylayers"]=fields["ylayers"]+pad_sides*2
    fields["xlayers"]=fields["xlayers"]+pad_top

    fields["h"]=fields["h"]+pad_top
    fields["l1"]=fields["l1"]+pad_sides
    fields["l2"]=fields["l2"]+pad_sides

    padding = ((pad_sides, pad_sides),(0, pad_top))    
    fields["phi"]=np.pad(
        fields["phi"],
        padding, 
        "constant", constant_values=(0.0, 0.0)
        )
    print("phi", "padded")


@memory.cache
def calculate_fields(
        a0, a1, 
        chi_PC, chi_PS, 
        wall_thickness, 
        pore_radius, 
        d,
        sigma,
        ###default parameters###
        exclude_volume = True, 
        truncate_pressure = False, 
        method = "convolve",
        mobility_correction = "vol_average",
        mobility_model = "Rubinstein",
        mobility_model_kwargs = {},
        partitioning_cutoff_phi = None,
        #Haberman_correction = False,
        stickiness = False,
        stickiness_model_kwargs = {},
        gel_phi = None,
        D_0 = "Einstein",
        linalg = True,
        linalg_kwargs = {"z_boundary":400}
        # pad_phi = None
        ):

    fields = utils.get_by_kwargs(master_empty, chi_PS = chi_PS, s = wall_thickness, r = pore_radius, sigma = sigma).squeeze()
    fields["phi"] = fields.dataset["phi"].squeeze().T
    # if pad_phi is not None:
    #     pad_phi_field(fields,*pad_phi)
    fields["d"] = d
    fields["chi_PC"] = chi_PC
    if gel_phi is not None:
        if not isinstance(gel_phi, numbers.Number):
            raise ValueError("phi_gel expected to be numeric")
        subst_by_gel(fields, phi= gel_phi)
    if d>=2:
        if np.isclose(d, np.round(d)):
            d = int(d)
        else:
            raise ValueError("Only even d is accepted")
        add_walls(fields, exclude_volume, d)
        method = method
        correct_excluded_volume = True
    else:#particle is very small
        add_walls(fields, exclude_volume = False, d=None)
        if method != "approx": print("small particle, approx free energy")
        method = "approx"
        mobility_correction = "none"
        correct_excluded_volume = False

    calculate_pressure(fields, truncate_pressure)
    calculate_gamma(fields, a0, a1, chi_PC)
    calculate_energy(fields, d, method)
    calculate_corrected_phi(fields, a0=a0, a1=a1, chi_PC = chi_PC, correction=mobility_correction, d=d)
    calculate_mobility(fields, d, 
                       mobility_model, 
                       mobility_model_kwargs, 
                       phi_arr="corrected_phi",
                       #phi_arr="phi",
                       #Haberman_correction = Haberman_correction, 
                       stickiness=stickiness, 
                       stickiness_model_kwargs=stickiness_model_kwargs
                       )
    calculate_conductivity(fields, D_0)
    # integrate_conductivity(fields, correct_excluded_volume=correct_excluded_volume)
    integrate_conductivity(fields)
    fields["permeability"] = fields["R"]**-1

    if partitioning_cutoff_phi is not None:
        calculate_partition_coefficient(fields, cutoff_phi = partitioning_cutoff_phi)
    
    D_0_ = fields["D_0"]
    fields["thin_empty_pore"] = empty_pore_permeability(D_0_, pore_radius-d/2, 0)
    fields["thick_empty_pore"] = empty_pore_permeability(D_0_, pore_radius-d/2, wall_thickness+d)
    fields["thick_empty_pore_Haberman"] = empty_pore_permeability_corrected(D_0_, pore_radius, wall_thickness, d)

    if linalg:
        from solve_poisson import R_solve
        R_solve(fields, **linalg_kwargs)

    return fields
# %%
if __name__ == "__main__":
    a0, a1 = 0.7, -0.3
    chi_PS = 0.5
    chi_PC = -1.0
    d=12
    L=52
    pore_radius = 26
    sigma = 0.02
    fields =  calculate_fields(a0, a1, chi_PC, chi_PS, L, pore_radius, d, sigma)
# %%
