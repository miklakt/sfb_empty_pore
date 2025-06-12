#%%
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
d= 12
chi_PS = [0.3, 0.9]
chi_PC = np.round(np.arange(0, -2.55, -0.1),3)
# model, mobility_model_kwargs = "Rubinstein", {"prefactor":30}

results = []
for chi_PS_, chi_PC_ in itertools.product(chi_PS, chi_PC):
    print(chi_PC_)
    result = calculate_fields(
        a0=a0, a1=a1, pore_radius=pore_radius, wall_thickness=wall_thickness,
        d=d, chi_PS=chi_PS_, chi_PC=chi_PC_,
        sigma = sigma,
        #exclude_volume=True,
        #truncate_pressure=False,
        method= "approx",
        #convolve_mode="same",
        #mobility_correction= "vol_average",
        #mobility_model = model,
        mobility_model_kwargs = {"prefactor": 30**(1/2)},
        linalg_kwargs={"z_boundary" : 300},
        gel_phi=0.3
        #integration="cylindrical_caps"
        )
        
    #result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
    results.append(result)
results = pd.DataFrame(results)
#%%
fig, ax = plt.subplots()

for chi_PS_, result in results.groupby(by = "chi_PS"):
    #R_int = result.R_pore
    #R_ext= result.R_left+result.R_right
    #R_tot = result.permeability**-1
    R_tot = result.R_lin_alg
    R_int = result.R_lin_alg_int#*1.057692305
    R_ext = result.R_lin_alg_ext# - R_int
    #R_ext= np.abs(result.R_lin_alg_ext)

    R_thin = 1/result.thin_empty_pore
    R_thick = 1/result.thick_empty_pore

    ax.plot(chi_PC, R_int, linestyle = "-.")
    color = ax.get_lines()[-1].get_color()
    ax.plot(chi_PC, R_ext, linestyle = "--", color = color, label = "R_ext")
    ax.plot(chi_PC, R_tot, label = chi_PS_,  color = color, linewidth =2)

ax.plot(chi_PC, R_thin, label = r"$R_{\text{ext}}^{0}$", color = "black", linewidth = 0.6)
ax.plot(chi_PC, R_thick, label = r"$R_{0}$", color = "black", linewidth = 0.6)
# ax.plot(chi_PC, R_int,  label = r"$R_{\text{int}}$")
# ax.plot(chi_PC, R_ext,  label = r"$R_{\text{ext}}$")
# ax.plot(chi_PC, R_ext+R_int, linestyle = "--", color = "k", label = r"$R_{\text{ext}}$")
# ax.plot(chi_PC, R_thin, label = r"$R_{\text{int}}^{0}$", color = "black", linewidth = 0.6)

ax.set_xlabel(r"$\chi_{\text{PC}}$", fontsize = 14)
ax.set_ylabel(r"$R \frac{k_\text{B} T}{\eta_{\text{S}}}$", fontsize = 14)
ax.set_yscale("log")
ax.set_xlim(-2.5,0)
ax.set_ylim(1e-1,1e3)
#ax.legend()

fig.set_size_inches(15,15)
fig.savefig("fig/resistance_comp_on_chi_PC.svg")
# %%
