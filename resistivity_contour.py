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
d = np.arange(2, 24, 2)
#d =[8 ,10, 12 ,]
chi_PS = [0.1, 0.3, 0.5]
chi_PC = np.round(np.arange(-2, -0.75, 0.1), 3)

# model, mobility_model_kwargs = "none", {}
# model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
# model = "Fox-Flory", dict(N = 300)
model, mobility_model_kwargs = "Rubinstein", {"prefactor":100}
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
        
    results.append(result)
results = pd.DataFrame(results)
results["log_d"] = np.log10(results["d"])
#%%
fig, axs = plt.subplots(ncols = len(chi_PS), sharey="row", nrows = 1, sharex = True)

for ax, (chi_PS_, result_) in zip(axs, results.groupby(by = "chi")):
    result__ = result_.pivot_table(values = "permeability", columns = "d", index = "chi_PC")
    result__2 = result_.pivot_table(values = "thick_empty_pore", columns = "d", index = "chi_PC")
    result__2 = result__/result__2
    extent = [min(result__.columns), max(result__.columns), min(result__.index), max(result__.index)]
    #ax.imshow(-np.log10(result__), extent = extent, origin = "lower", interpolation = "bilinear", vmax = 3, cmap = "cividis_r")
    #ax.set_xscale("log")
    cs = ax.contour(-np.log10(result__), 
                    extent = extent, 
                    origin = "lower", 
                    colors = "black", 
                    levels = np.round(np.arange(-0.25, 3, 0.25), 3),
                    linewidths = 0.2
                    )
    #ax.clabel(cs, inline = False)
    cs = ax.contour(-np.log10(result__), extent = extent, origin = "lower", colors = "black", levels = np.arange(0, 5))
    ax.clabel(cs, inline = True, fmt = "$10^{{{%d}}}$")
    cs = ax.contour(-np.log10(result__), extent = extent, origin = "lower", colors = "black", levels = np.arange(4, 12, 2))
    ax.clabel(cs, inline = True, fmt = "$10^{{{%d}}}$")
    cs = ax.contour(-np.log10(result__), extent = extent, origin = "lower", colors = "black", levels = np.arange(10, 25, 5))
    ax.clabel(cs, inline = True, fmt = "$10^{{{%d}}}$")

    #ax.imshow(result__2, extent = extent, origin = "lower", cmap = "Greens", interpolation = "bicubic", vmin = 1, vmax = 1e2)

    ax.contour(result__2, extent = extent, origin = "lower", colors = "red", levels = [1])
    #ax.clabel(cs, inline = False)

    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    ax.set_xlabel("$d$")

    ax.set_xticks(np.arange(4, 24, 2), minor=True)
    ax.set_yticks(np.arange(-2, -1, 0.1), minor=True)

axs[0].set_ylabel("$\chi_{PC}$")
axs[1].plot([],[], color = "red", label = r"$R = R^{\textrm{empty}}$")
axs[1].legend(loc ="lower left")
fig.set_size_inches(6, 3)
plt.tight_layout()
#fig.savefig("fig/resistivity_contourplot.svg")
#fig.savefig("tex/third_report/fig/resistivity_contourplot_low_d.svg")
# %%
