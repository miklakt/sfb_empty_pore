# %%
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rc

rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')
style.use('tableau-colorblind10')
mpl_markers = ('o', '+', 'x', 's', 'D')

from calculate_fields_in_pore import *

a0 = 0.70585835
a1 = -0.31406453
wall_thickness=52
pore_radius=26
sigma = 0.02

#%%
#d = np.arange(6, 24, 2)
d_color = [8, 12, 16, 20]
d = d_color
chi_PS = [0.1, 0.3, 0.5]
chi_PC = np.round(np.arange(-2, 0.2, 0.05),3)

# model, mobility_model_kwargs = "none", {}
# model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
# model = "Fox-Flory", dict(N = 300)
model, mobility_model_kwargs = "Rubinstein", {"prefactor":1}
#model, mobility_model_kwargs = "Hoyst", {"alpha" : 1.63, "delta": 0.89, "N" : 300}

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
        cutoff_phi=1e-4,
        )
    
    #result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
    
    results.append(result)
results = pd.DataFrame(results)

#%%
fig, axs = plt.subplots(ncols = len(chi_PS))
results_ = results.loc[(results.mobility_model == model)]

permeability_field = "permeability"
for ax, (chi_PS_, result_) in zip(axs, results_.groupby(by = "chi")):
    markers = itertools.cycle(mpl_markers)
    for d_, result__ in result_.groupby(by = "d"):
        x = result__["PC"].squeeze()
        y = result__[permeability_field].squeeze()#*d_
        if d_ in d_color:
            plot_kwargs = dict(
                label = fr"$d = {d_}$",
                marker = next(markers),
                markevery = 0.5,
                markersize = 4,
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

    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    ax.set_ylim(1e-6, 1e1)
    ax.set_xlim(1e-1,1e5)
    ax.set_xlabel("$c/c_0$")
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.grid()

    ax.axvline(1, color = "red", linewidth = 0.5)


axs[0].set_ylabel(r"$P \cdot \frac{\eta_0}{k_B T} $")

#axs[-1].scatter([],[], marker = "*", color = "grey", label = "empty pore")
axs[-1].legend( 
    #bbox_to_anchor = [1.0, -0.05],
    loc = "lower right"
    )

plt.tight_layout()
fig.set_size_inches(9, 3)
#%%
fig, axs = plt.subplots(ncols = len(chi_PS))
results_ = results.loc[(results.mobility_model == model)]

for ax, (chi_PS_, result_) in zip(axs, results_.groupby(by = "chi")):
    markers = itertools.cycle(mpl_markers)
    for d_, result__ in result_.groupby(by = "d"):
        x = result__["chi_PC"].squeeze()
        y = result__["permeability"].squeeze()#*d_
        if d_ in d_color:
            plot_kwargs = dict(
                label = fr"$d = {d_}$",
                marker = next(markers),
                markevery = 0.5,
                markersize = 4,
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

    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    ax.set_ylim(1e-6, 1e1)
    #ax.set_xlim(1e-1,1e5)
    ax.set_xlabel("$\chi_{PC}$")
    ax.set_yscale("log")
    #ax.set_xscale("log")

    ax.grid()

    #ax.axvline(-1, color = "red", linewidth = 0.5)


axs[0].set_ylabel(r"$P \cdot \frac{\eta_0}{k_B T} $")

#axs[-1].scatter([],[], marker = "*", color = "grey", label = "empty pore")
axs[0].legend( 
    #bbox_to_anchor = [1.0, -0.05],
    loc = "upper right"
    )

plt.tight_layout()
fig.set_size_inches(9, 3)
# %%
d = [8, 12, 16]
#chi_PS = np.round(np.arange(0,1,0.1),2)
chi_PS = [0.1, 0.3, 0.5, 0.7]
chi_PS_color = chi_PS
chi_PC = np.round(np.arange(-2, 0.2, 0.05),3)
L=20

# model, mobility_model_kwargs = "none", {}
# model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
# model = "Fox-Flory", dict(N = 300)
model, mobility_model_kwargs = "Rubinstein", {"prefactor":1}
#model, mobility_model_kwargs = "Hoyst", {"alpha" : 1.63, "delta": 0.89, "N" : 300}

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
        cutoff_phi=1e-4,
        )
    
    #result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
    
    results.append(result)
results = pd.DataFrame(results)
#%%
fig, axs = plt.subplots(ncols = len(d))
results_ = results.loc[(results.mobility_model == model)]

permeability_field = "permeability"
for ax, (d_, result_) in zip(axs, results_.groupby(by = "d")):
    markers = itertools.cycle(mpl_markers)
    for chi_PS_, result__ in result_.groupby(by = "chi"):
        x = result__["PC"].squeeze()
        y = result__[permeability_field].squeeze()#*d_
        if chi_PS_ in chi_PS_color:
            plot_kwargs = dict(
                label = fr"$\chi_{{PS}} = {chi_PS_}$",
                marker = next(markers),
                markevery = 0.5,
                markersize = 4,
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

    ax.set_title(f"$d = {d_}$")
    ax.set_ylim(1e-6, 1e1)
    ax.set_xlim(1e-1,1e5)
    ax.set_xlabel("$c/c_0$")
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.grid()

    ax.axvline(1, color = "red", linewidth = 0.5)


axs[0].set_ylabel(r"$P \cdot \frac{\eta_0}{k_B T} $")

#axs[-1].scatter([],[], marker = "*", color = "grey", label = "empty pore")
axs[0].legend( 
    #bbox_to_anchor = [1.0, -0.05],
    loc = "lower right"
    )

plt.tight_layout()
fig.set_size_inches(6, 2.5)
plt.tight_layout()
#fig.savefig("tex/third_report/fig/permeability_on_partition.svg", dpi =600)
# %%
