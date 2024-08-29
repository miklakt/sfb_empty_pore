# %%
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rc

#rc('text',usetex=True)
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
#%%
d_color= [4, 8, 16]
d = [2,4,6,8,10,12,14]
#d =[8 ,10, 12 ,]
chi_PS = [0.1, 0.3, 0.5]
#chi_PC = [-2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1, -0.75]
chi_PC = np.round(np.arange(0, -1.6, -0.1),3)

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
        convolve_mode="same",
        mobility_correction= "vol_average",
        mobility_model = model,
        mobility_model_kwargs = mobility_model_kwargs,
        integration="cylindrical_caps"
        )
        
    result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
    results.append(result)
results = pd.DataFrame(results)

#%%

fig, axs = plt.subplots(ncols = len(chi_PS), sharey="row", nrows = 1, sharex = True)
results_ = results.loc[(results.mobility_model == model)]

for ax, (chi_PS_, result_) in zip(axs, results_.groupby(by = "chi")):
    markers = itertools.cycle(mpl_markers)
    for d_, result__ in result_.groupby(by = "d"):
        x = result__["chi_PC"].squeeze()
        y = 1/result__["permeability"]

        # y = np.gradient(np.log(y))/np.gradient(np.log(x))
        # y = np.gradient(y)/np.gradient(np.log(x))

        if d_ in d_color:
            plot_kwargs = dict(
                label = fr"$d = {d_}$",
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

    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    ax.set_ylim(1e-1, 1e3)
    #ax.set_ylim(-5, 20)
    ax.set_yscale("log")
    #ax.set_xscale("log")

axs[0].legend()

axs[0].set_ylabel(r"$R \cdot \frac{k_B T}{\eta_0}$")
# axs[1,-1].plot([],[], color = "black", linestyle = ":", linewidth = 2,

#                 label = "$R_{conv}$"
#                 )

# axs[1,-1].plot([],[], color = "black", linestyle = "-.", linewidth = 2,
#                 label = "$R_{channel}$"
#                 )

# axs[1,-1].plot([],[], color = "black", linestyle = "-", linewidth = 2,
#                 label = "$R_{empty}$"
#                 )
fig.supxlabel("$\chi_{PS}$")
    #axs[1,-1].legend()

#plt.tight_layout()
#fig.set_size_inches(7, 7)
#fig.set_size_inches(7, 2.5)
#fig.savefig("fig/permeability_on_d.svg")
#fig.savefig("tex/third_report/fig/permeability_on_d_detailed_low_d.svg")
# %%
