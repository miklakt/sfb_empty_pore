# %%
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rc

# rc('text',usetex=True)
# rc('text.latex', preamble=r'\usepackage{color}')
style.use('tableau-colorblind10')


from calculate_fields_in_pore import *
import scf_pb

a0 = 0.70585835
a1 = -0.31406453
wall_thickness=52
pore_radius=26
sigma = 0.02

def free_energy_gel(phi, chi_PS, chi_PC, d):
    V = volume(d)
    S = surface(d)
    Pi_ = Pi(phi, chi_PS, trunc=False)
    gamma_ = gamma(chi_PS, chi_PC, phi, a0, a1)
    free_energy = Pi_*V + gamma_*S
    return free_energy

def PC_gel(phi, chi_PS, chi_PC, d):
    FE = free_energy_gel(phi, chi_PS, chi_PC, d)
    PC = np.exp(-FE)
    return PC
#%%
#d = np.arange(6, 24, 2)
d_color = [4, 6, 8]
d = d_color
chi_PS = [0.3, 0.5]
chi_PC = np.round(np.arange(-3, 0.2, 0.05),3)

# model, mobility_model_kwargs = "none", {}
# model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
# model = "Fox-Flory", dict(N = 300)
model, mobility_model_kwargs = "Rubinstein", {"prefactor":1}
#model, mobility_model_kwargs = "Hoyst", {"alpha" : 1.63, "delta": 0.89, "N" : 300}
Haberman_correction_ = False
results = []
for d_, chi_PS_, chi_PC_ in itertools.product(d, chi_PS, chi_PC):
    print(d_, chi_PS_, chi_PC_)
    result = calculate_permeability(
        a0, a1, pore_radius, wall_thickness,
        d_, chi_PS_, chi_PC_,
        sigma = sigma,
        exclude_volume=True,
        truncate_pressure=False,
        method= "convolve",
        convolve_mode="valid",
        mobility_correction= "vol_average",
        mobility_model = model,
        mobility_model_kwargs = mobility_model_kwargs,
        integration="cylindrical_caps",
        cutoff_phi=1e-2,
        Haberman_correction=Haberman_correction_
        )
    
    #result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
    
    results.append(result)
results = pd.DataFrame(results)
#%%
experimental_data = pd.read_csv("npc_permeation_probes.csv")
#%%
def create_interp_func(X, Y, domain=None):
    from scipy.interpolate import CubicSpline
    """
    Create an interpolation function using cubic splines.

    Parameters:
        X (array-like): Array of X values.
        Y (array-like): Array of Y values corresponding to X.
        domain (tuple, optional): A tuple (new_min, new_max) defining the new domain for remapping. Default is None.

    Returns:
        function: An interpolation function interp_func(x).
    """
    # Ensure X and Y are numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Create the cubic spline interpolation function
    spline = CubicSpline(X, Y)

    # Define the interpolation function with optional domain remapping
    def interp_func(x):
        # Remap x if a domain is provided
        if domain is not None:
            new_min, new_max = domain
            old_min, old_max = np.min(X), np.max(X)
            x = new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)
        return spline(x)

    return interp_func

#%%
reference_chi_PC = -0.0
Kuhn_segment = 0.8
chi_gel = 0.5

phi_gel = 0.2
results["PC_gel"]=results.apply(lambda _:PC_gel(phi_gel, chi_gel, _["chi_PC"], _["d"]), axis = 1)
#%%
def get_permeability_different_bc(boundary_z, permeability):
    R_reservoir = (-np.log(pore_radius/3 + boundary_z) + np.log(pore_radius + boundary_z))/(2*np.pi*pore_radius)
    return (permeability**(-1)-R_reservoir)**(-1)

def get_permeability_longer_pore(add_length, permeability_z, permeability):
    R_added  = (permeability_z[int(len(permeability_z)/2)])**(-1)*add_length
    return (permeability**(-1)+R_added)**(-1)


add_length=150
results["permeability_longer"]=results.apply(lambda _:get_permeability_longer_pore(add_length, _.permeability_z, _.permeability), axis = 1)

results["permeability_channel"] = results["R_pore"]**(-1)
# boundary_z = 0
# results["permeability_bc"]=results.apply(lambda _:get_permeability_different_bc(boundary_z, _.permeability), axis = 1)
#%%

################################################################################
################################################################################
fig, axs = plt.subplots(ncols = len(chi_PS), dpi = 600, sharey=True, sharex= True)
if len(chi_PS) == 1:
    axs_ = [axs]
else:
    axs_ = axs
results_ = results.loc[(results.mobility_model == model)]

for ax, (chi_PS_, result_) in zip(axs_, results_.groupby(by = "chi")):
    x = experimental_data["Passage_Rate"]
    mpl_markers = ('o', 's', 'D')
    markers = itertools.cycle(mpl_markers)
    for nup in ["Mac98A","Nup116","Nsp1"]:
    #for nup in ["Nup116"]:
        y = experimental_data[nup]
        ax.scatter(x,y, marker = next(markers), s=10, label = nup, color = "black", fc = "none")

    markers = itertools.cycle(mpl_markers)
    for nup in ["Mac98A","Nup116","Nsp1"]:
        x = experimental_data.loc[experimental_data.Protein == "mCherry","Passage_Rate"]
        y = experimental_data.loc[experimental_data.Protein == "mCherry",nup]
        ax.scatter(x,y, color = "red", marker = next(markers), s =10)

    for d_, result__ in result_.groupby(by = "d"):
        permeability_key = "permeability_channel"
        reference_permeability = result__.loc[result__.chi_PC==reference_chi_PC,permeability_key].squeeze()
        x = result__[permeability_key].squeeze()#*d_
        #y = result__["PC"].squeeze()
        y = result__["PC_gel"].squeeze()
        x=x/reference_permeability
        #x=reference_permeability/x
        ax.plot(
            x, y, 
            label = fr"$d = {d_}({d_*Kuhn_segment:.1f}"+r"\text{nm})$",
            #marker = next(markers),
            #markevery = 0.5,
            #markersize = 4,
        )
        permeability_key = "permeability"
        reference_permeability = result__.loc[result__.chi_PC==reference_chi_PC,permeability_key].squeeze()
        x = result__[permeability_key].squeeze()#*d_
        #y = result__["PC"].squeeze()
        y = result__["PC_gel"].squeeze()
        x=x/reference_permeability
        ax.plot(x, y, linestyle = "--", color = ax.lines[-1].get_color(), linewidth= 0.5)

        permeability_key = "permeability_longer"
        reference_permeability = result__.loc[result__.chi_PC==reference_chi_PC,permeability_key].squeeze()
        x = result__[permeability_key].squeeze()#*d_
        #y = result__["PC"].squeeze()
        y = result__["PC_gel"].squeeze()
        x=x/reference_permeability
        ax.plot(x, y, linestyle = "--", color = ax.lines[-1].get_color())

    ax.set_title(r"$\chi_{{PS}} = "+f"{chi_PS_}$")
    ax.set_xlim(1e-1, 1e4)
    ax.set_ylim(1e-2,1e4)
    ax.set_xlabel(r"$R_{\text{reference}} / R$")
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.grid()

    #ax.axvline(1, color = "red", linewidth = 0.5)


axs_[0].set_ylabel(r"$c_{\text{eq}}/c_0$")

#axs_[-1].scatter([],[], marker = "*", color = "grey", label = "empty pore")
axs_[-1].legend( 
    bbox_to_anchor = [1.05, 0.05],
    loc = "lower left"
    )

#plt.tight_layout()
fig.set_size_inches(2.5*len(axs_)+0.5, 3)
#fig.savefig()
#%%