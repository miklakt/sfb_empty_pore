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

a0 = 0.70585835
a1 = -0.31406453
wall_thickness=52
pore_radius=26
sigma = 0.02

#%%
#d = np.arange(6, 24, 2)
d_color = [6, 8]
d = d_color
chi_PS = [0.3, 0.5]
chi_PC = np.round(np.arange(-3, 0.2, 0.05),3)

# model, mobility_model_kwargs = "none", {}
# model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
# model = "Fox-Flory", dict(N = 300)
model, mobility_model_kwargs = "Rubinstein", {"prefactor":1}
#model, mobility_model_kwargs = "Hoyst", {"alpha" : 1.63, "delta": 0.89, "N" : 300}
Haberman_correction_ = True
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
################################################################################
################################################################################
fig, axs = plt.subplots(ncols = len(chi_PS), dpi = 600, sharey=True, sharex= True)
if len(chi_PS) == 1:
    axs_ = [axs]
else:
    axs_ = axs
results_ = results.loc[(results.mobility_model == model)]

reference_chi_PC = -0.5

Kuhn_segment = 0.8
for ax, (chi_PS_, result_) in zip(axs_, results_.groupby(by = "chi")):
    x = experimental_data["Passage_Rate"]
    mpl_markers = ('o', 's', 'D')
    markers = itertools.cycle(mpl_markers)
    for nup in ["Mac98A","Nup116","Nsp1"]:
        y = experimental_data[nup]
        ax.scatter(x,y, marker = next(markers), s=10, label = nup, color = "black", fc = "none")

    markers = itertools.cycle(mpl_markers)
    for nup in ["Mac98A","Nup116","Nsp1"]:
        x = experimental_data.loc[experimental_data.Protein == "mCherry","Passage_Rate"]
        y = experimental_data.loc[experimental_data.Protein == "mCherry",nup]
        ax.scatter(x,y, color = "red", marker = next(markers), s =10)

    for d_, result__ in result_.groupby(by = "d"):
        reference_permeability = result__.loc[result__.chi_PC==reference_chi_PC,"permeability"].squeeze()
        x = result__["permeability"].squeeze()#*d_
        y = result__["PC"].squeeze()
        x=x/reference_permeability
        if d_ in d_color:
            plot_kwargs = dict(
                label = fr"$d = {d_}({d_*Kuhn_segment:.1f}"+r"\text{nm})$",
                #marker = next(markers),
                #markevery = 0.5,
                #markersize = 4,
            )
        else:
            plot_kwargs = dict(
                linewidth = 0.1,
                color ="black"
            )
        ax.plot(
            x, y, 
            **plot_kwargs
            )
        # if d_ in d_color:
        #     ax.scatter(
        #         1, 
        #         result__["thick_empty_pore"].iloc[1],#*d_, 
        #         marker = "*"
        #         )
            #ax.plot(x, result__["thick_empty_pore"], linestyle = "--", color = ax.lines[-1].get_color())

    ax.set_title(r"$\chi_{{PS}} = "+f"{chi_PS_}$")
    ax.set_xlim(1e-1, 1e3)
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