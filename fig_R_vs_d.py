# Pore resistance R, normalized by the viscosity of the solvent
# as a function of colloid size d 
# (a) for selected polymer-colloid interaction
# strengths at a fixed solvent strength
# (b) for selected solvent strengths (χPS, as indicated with colored lines) at a
# fixed polymer-colloid interaction strengths χPC = −1.25.
# The normalized resistance of an empty pore R0 
# (black thick lines) serves as a reference.
#%%
import itertools
import functools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rc


style.use('tableau-colorblind10')
get_palette_colors = lambda: itertools.chain([
 '#FF800E',
 '#006BA4',
 '#ABABAB',
 '#595959',
 '#5F9ED1',
 '#C85200',
 '#898989',
 '#A2C8EC',
 '#FFBC79',
 '#CFCFCF'])
mpl_markers = ('+','o', '^', 's', 'D')

import calculate_fields_in_pore

a0 = 0.7
a1 = -0.3
L=52
r_pore=26
sigma = 0.02
alpha =  30**(1/2)
d = np.arange(2.0, 32.0, 2)
d = np.insert(d, 0, [0.5, 1])

#%%
calculate_fields = functools.partial(
    calculate_fields_in_pore.calculate_fields,
        a0=a0, a1=a1, 
        wall_thickness = L, 
        pore_radius = r_pore,
        sigma = sigma,
        mobility_model_kwargs = {"prefactor":alpha},
    )

R_0_no_vol_excl = np.array([calculate_fields_in_pore.empty_pore_permeability(1/(3*np.pi*d_), r_pore, L)**-1 for d_ in d])
R_0 = np.array([calculate_fields_in_pore.empty_pore_permeability(1/(3*np.pi*d_), r_pore-d_/2, L+d_)**-1 for d_ in d])
#%%
fig, axs = plt.subplots(nrows = 2, 
                        sharex = False)

chi_PS = 0.5
chi_PCs =[-1.25] + [0,-0.5, -1.0, -1.5]
ax = axs[0]
marker = itertools.chain(mpl_markers)
color = get_palette_colors()
for chi_PC in chi_PCs:
    calc = pd.DataFrame([calculate_fields(chi_PC=chi_PC, chi_PS=chi_PS, d = d_) for d_ in d])
    x=d
    y = calc["permeability"]**-1
    ax.plot(
        x, y, 
        linewidth = 0.5,
        ms=4,
        marker = next(marker),
        color = next(color),
        label = chi_PC,
        )
ax.text(0.05, 0.95, r"$\chi_{\text{PS}} = "+f"{chi_PS}$", transform = ax.transAxes, va = "top", ha = "left", bbox ={"fc" : "white"})
ax.legend(bbox_to_anchor = [1, 1], title = r"$\chi_{\text{PC}}$")

ax = axs[1]
chi_PC = -1.25
chi_PSs = [ 0.5]+[0.3, 0.4, 0.6, 0.7]
marker = itertools.chain(mpl_markers)
color = get_palette_colors()
for chi_PS in chi_PSs:
    calc = pd.DataFrame([calculate_fields(chi_PC=chi_PC, chi_PS=chi_PS, d = d_) for d_ in d])
    x=d
    y = calc["permeability"]**-1
    ax.plot(
        x, y, 
        linewidth = 0.5,
        ms=4,
        mfc = "none",
        marker = next(marker),
        color = next(color),
        label = chi_PS
        )
ax.text(0.05, 0.95, r"$\chi_{\text{PC}} = "+f"{chi_PC}$", transform = ax.transAxes, va = "top", ha = "left", bbox ={"fc" : "white"})

for ax in axs:
    ax.plot(x, R_0, color = "k", linewidth=2)
    ax.plot(x, R_0_no_vol_excl, color = "k", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e3)
    ax.set_xlim(5e-1, 32)
    ax.grid()

axs[1].set_xlabel("$d$",fontsize = 14)
axs[0].set_ylabel(r"$R \frac{k_{\text{B}}T}{\eta_{\text{S}}}$", fontsize = 14)
axs[1].set_ylabel(r"$R \frac{k_{\text{B}}T}{\eta_{\text{S}}}$", fontsize = 14)

ax.legend(bbox_to_anchor = [1, 1], title = r"$\chi_{\text{PC}}$")

fig.set_size_inches(2.2,5.5)
fig.savefig("fig/permeability_on_d.svg")

# %%
