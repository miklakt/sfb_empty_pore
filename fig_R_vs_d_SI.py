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
from matplotlib import rc, rcParams

rcParams.update({
    "mathtext.fontset": "cm",  # Use Computer Modern
    "font.family": "serif",
})

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
mpl_markers = ('+','o', '^', 's', 'D', 'd')

import calculate_fields_in_pore

a0 = 0.7
a1 = -0.3
L=52
r_pore=26
sigma = 0.02
alpha =  30**(1/2)
d = np.arange(2.0, 32.0, 2)
d = np.insert(d, 0, [0.5, 1])

show_simulation_results = True
#%%
def correct_flux(J, d, pore_radius=26, wall_thickness=52, ylayers=492, l1=220):
    #as the simulation box is finite, it has lower resistance than an infinite reservoir
    z_left = l1-d/2
    z_right = ylayers-l1-wall_thickness+d
    pore_radius_ = pore_radius-d/2
    R_left = (np.pi - 2*np.arctan(z_left/pore_radius_))/(4*np.pi*pore_radius_)*np.pi
    R_right = (np.pi - 2*np.arctan(z_right/pore_radius_))/(4*np.pi*pore_radius_)*np.pi
    J_corrected = 1/(1/J + R_left + R_right)
    return J_corrected

simulation_results = pd.read_csv("numeric_simulation_results_.csv")
simulation_results["J_corrected"] = correct_flux(simulation_results["J_tot"],simulation_results["d"])
simulation_results["R"] = 1/(simulation_results["J_tot"]/simulation_results["d"]/3)
simulation_results["R_corrected"] = 1/(simulation_results["J_corrected"]/simulation_results["d"]/3)
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
fig, ax = plt.subplots()

chi_PS = 0.5
chi_PCs =[-1.3] + [0.0, -1.0, -1.4, -1.8]
marker = itertools.chain(mpl_markers)
color = get_palette_colors()
exp_data_marker = "s"
for chi_PC in chi_PCs:
    calc = pd.DataFrame([calculate_fields(chi_PC=chi_PC, chi_PS=chi_PS, d = d_) for d_ in d])
    x = d
    y = calc["permeability"]**-1/d
    color_ = next(color)
    ax.plot(
        x, y, 
        linewidth = 1.5 if chi_PC==-1.3 else 0.7,
        ms=3,
        marker = next(marker),
        color = color_,
        label = chi_PC,
        )
    if show_simulation_results:
        sim_data = simulation_results.query(f"chi_PS == {chi_PS} & chi_PC == {chi_PC}")
        if not sim_data.empty:
            x = sim_data["d"]
            y = sim_data["R"]/x
            ax.plot(
                x, y, 
                ms=7,
                #mec="k",
                marker = exp_data_marker,
                color = color_,
                #label = chi_PC,
                mfc = "none",
                linewidth = 0
                )

# Get handles and labels
handles, labels = ax.get_legend_handles_labels()

# Sort by labels in descending order
sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: float(x[1]), reverse=True)
sorted_handles, sorted_labels = zip(*sorted_handles_labels)
legend_1 = fig.legend(sorted_handles, sorted_labels, 
          bbox_to_anchor = [0.98, 0.95],
          loc='upper left',
          title = r"$\chi_{\text{PC}}$",
          title_fontsize = 12,
          )
#fig.add_artist(legend_1) 
ax.text(0.02, 0.98, r"$\chi_{\text{PS}} = "+f"{chi_PS}$", 
        transform = ax.transAxes, va = "top", ha = "left", 
        bbox ={"fc" : "white", "pad":1},
        fontsize = 12,
        )


ax.plot(x, R_0/d, color = "k", linewidth=2)
ax.plot(x, R_0_no_vol_excl/d, color = "k", linewidth=0.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-2, 1e3)
ax.set_xlim(5e-1, 32)
ax.grid()
ax.text(x[-8], R_0[-8]/x[-8], r"$R_0$", 
    #transform = ax.transAxes, 
    va = "center", ha = "center", 
    bbox ={"fc" : "white", "ec":"none", "pad":0.2},
    fontsize = 12,
    rotation = 22,
    )

ax.set_xlabel("$d$",fontsize = 14,labelpad=-5)
ax.set_ylabel(r"$\; \frac{R}{d} \, \frac{k_{\text{B}}T}{\eta_{\text{S}}}$", fontsize = 16, labelpad=-7)

dummy_line, = ax.plot([],[], 
                      marker=exp_data_marker, 
                      linewidth = 0, 
                      color = "k", 
                      mfc = "none",
                      label = "numerical")
second_legend = fig.legend([dummy_line], ["numerical"], 
          bbox_to_anchor = [0.12, 0.25],
          loc='upper left',
          )



plt.tight_layout()
fig.set_size_inches(3.2,3)
fig.savefig("fig/R_vs_d_SI.svg")
#fig.savefig("tex/fig/R_vs_d_SI.svg")
# %%
