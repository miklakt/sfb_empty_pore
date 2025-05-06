# %%
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rc

#rc('text',usetex=True)
#rc('text.latex', preamble=r'\usepackage{color}')
style.use('tableau-colorblind10')
mpl_markers = ('o', '+', 'x', 's', 'D')

from calculate_fields_in_pore import *

a0 = 0.70585835
a1 = -0.31406453
wall_thickness=52
pore_radius=26
sigma = 0.02
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

def flux_to_an_adsorbing_dome(r_dome):
    return 4*np.pi*r_dome

def get_dome_permeability(d, r_dome = 26*1.1):
    return np.pi*r_dome/(3*np.pi*d)

simulation_results = pd.read_csv("numeric_simulation_results_.csv")
simulation_results["J_corrected"] = correct_flux(simulation_results["J_tot"],simulation_results["d"])
simulation_results["R"] = 1/(simulation_results["J_tot"]/simulation_results["d"]/3)
simulation_results["R_corrected"] = 1/(simulation_results["J_corrected"]/simulation_results["d"]/3)
#simulation_results["J_corrected"] = correct_the_flux(simulation_results.J_tot, simulation_results.d)
#%%
simulation_empty_pore = pd.DataFrame(
    columns = ["d", "J_tot"],
    data = dict(
            d=[4, 8, 16, 24],
            #R = [0.203, 0.302]
            J_tot = [5.927, 4.926, 3.311, 2.063]
        )
)
simulation_empty_pore["J_corrected"] = correct_flux(simulation_empty_pore["J_tot"],simulation_empty_pore["d"])
simulation_empty_pore["R"] = 1/(simulation_empty_pore["J_tot"]/simulation_empty_pore["d"]/3)
simulation_empty_pore["R_corrected"] = 1/(simulation_empty_pore["J_corrected"]/simulation_empty_pore["d"]/3)

#%%
d = np.arange(2.0, 32.0, 2)
d = np.insert(d, 0, [0.5, 1])
#d =[8 ,10, 12 ,]
chi_PS = [0.3, 0.4, 0.5, 0.6]
#chi_PC = [-2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1, -0.75]
chi_PC_color = [-1.5, -1.25, -1.0, -0.5, 0]
chi_PC = chi_PC_color

# model, mobility_model_kwargs = "none", {}
# model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
# model = "Fox-Flory", dict(N = 300)
model, mobility_model_kwargs = "Rubinstein", {"prefactor":30}
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
        convolve_mode="same",
        mobility_correction= "vol_average",
        mobility_model = model,
        mobility_model_kwargs = mobility_model_kwargs,
        integration="cylindrical_caps",
        Haberman_correction=Haberman_correction_,
        stickiness=False,
        #gel_phi = 0.5
        )
        
    #result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
    results.append(result)
results = pd.DataFrame(results)

results_no_fe = []
for d_, chi_PS_, chi_PC_ in itertools.product(d, chi_PS, chi_PC):
    print(d_, chi_PS_, chi_PC_)
    result_no_fe = calculate_permeability(
        a0, a1, pore_radius, wall_thickness,
        d_, chi_PS_, chi_PC_,
        sigma = sigma,
        exclude_volume=True,
        truncate_pressure=False,
        method= "no_free_energy",
        convolve_mode="same",
        mobility_correction= "vol_average",
        mobility_model = model,
        mobility_model_kwargs = mobility_model_kwargs,
        # mobility_model = "Phillies",
        # mobility_model_kwargs = {"beta" : 10, "nu":0.76},
        integration="cylindrical_caps",
        Haberman_correction=Haberman_correction_,
        stickiness=False,
        )
        
    #result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
    results_no_fe.append(result_no_fe)
results_no_fe = pd.DataFrame(results_no_fe)

# results_no_D = []
# for d_, chi_PS_, chi_PC_ in itertools.product(d, chi_PS, chi_PC):
#     print(d_, chi_PS_, chi_PC_)
#     result_no_D = calculate_permeability(
#         a0, a1, pore_radius, wall_thickness,
#         d_, chi_PS_, chi_PC_,
#         sigma = sigma,
#         exclude_volume=True,
#         truncate_pressure=False,
#         method= "convolve",
#         convolve_mode="same",
#         mobility_correction= "vol_average",
#         mobility_model = "Phillies",
#         mobility_model_kwargs = {"beta" : 10, "nu":0.76},
#         integration="cylindrical_caps",
#         Haberman_correction=Haberman_correction_,
#         stickiness=False,
#         )
        
#     #result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
#     results_no_D.append(result_no_D)
# results_no_D = pd.DataFrame(results_no_D)
#         mobility_correction= "vol_average",
#         mobility_model = "Phillies",
#         mobility_model_kwargs = {"beta" : 10, "nu":0.76},
#         integration="cylindrical_caps",
#         Haberman_correction=Haberman_correction_,
#         stickiness=False,
#         )
        
#     #result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
#     results_no_D.append(result_no_D)
# results_no_D = pd.DataFrame(results_no_D)

#%%
show_contributions = False
show_CFD = False
show_analytical = True
ncols = 3 
#ncols=len(chi_PS)

fig, axs = plt.subplots(ncols = ncols, 
                        #sharey="row", 
                        nrows = 1, 
                        sharex = True
                        )
first_row_axes = axs
results_ = results.loc[(results.mobility_model == model) & (results.chi.isin([0.4, 0.5, 0.6]))]

mpl_markers = ('o', '+', 'x', 's', 'D')
for ax, (chi_PS_, result_) in zip(first_row_axes, results_.groupby(by = "chi")):
    markers = itertools.cycle(mpl_markers)
    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    for chi_PC_, result__ in result_.groupby(by = "chi_PC"):
        x = result__["d"].squeeze()
        #y = 1/result__["permeability"]/x
        y = 1/result__["permeability"]


        if chi_PC_ in chi_PC_color:
            plot_kwargs = dict(
                #label = fr"$\chi_{{PC}} = {chi_PC_}$",
                label = fr"${chi_PC_:.2f}$",
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
            **plot_kwargs,
            marker = next(markers),
            #mfc = "none",
            ms = 3,
            linewidth = 0.2
            )

        if show_CFD:
            R = simulation_results.query(f"(chi_PS=={chi_PS_})&(chi_PC=={chi_PC_})")
            ax.scatter(
                R["d"], 
                #R["R_corrected"]/R["d"],
                R["R_corrected"],
                color = ax.lines[-1].get_color(),
                marker = "s",
                facecolor = "none",
                s=20,
                linewidth = 0.5
                )

    if show_analytical:
        if Haberman_correction_:
            ax.plot(
                d, 
                #1/result__["thick_empty_pore_Haberman"]/d, 
                1/result__["thick_empty_pore_Haberman"],
                color = "black", 
                linestyle = "-",
                label = r"$R^{\ast}_{0}$",
                linewidth = 2,
                zorder = -1,
                )
        else:
            ax.plot(
                d, 
                #1/result__["thick_empty_pore"]/d,
                1/result__["thick_empty_pore"],
                color = "black", 
                linestyle = "-",
                label = "$R_{0}$",
                linewidth = 2,
                zorder = -1,
                )

    if show_CFD:
        ax.scatter(
            simulation_empty_pore["d"], 
            #simulation_empty_pore["R_corrected"]/simulation_empty_pore["d"],
            simulation_empty_pore["R_corrected"],
            color = "black",
            facecolor = "none",
            marker = "s",
            label = "numerical\n simulation",
            s=20,
            linewidth = 0.5
            )

    if chi_PC_ not in [-1.0, -1.25, -1.5]: continue
    markers = itertools.cycle(mpl_markers)
    for chi_PS_, result__ in result_no_fe_.groupby(by = "chi"):
        x = result__["d"].squeeze()
        #y = 1/result__["permeability"]/x
        y = 1/result__["permeability"]

        plot_kwargs = dict(
            label = fr"${chi_PS_:.2f}$ no FE",
        )

        ax.plot(
            x, y, 
            **plot_kwargs,
            marker = next(markers),
            mfc = "none",
            ms = 3,
            linewidth = 0.2,
            color = "k"
            )
    
        if show_analytical:
            if Haberman_correction_:
                ax.plot(
                    d, 
                    #1/result__["thick_empty_pore_Haberman"]/d, 
                    1/result__["thick_empty_pore_Haberman"],
                    color = "black", 
                    linestyle = "-",
                    label = r"$R^{\ast}_{0}$",
                    linewidth = 2,
                    zorder = -1,
                    )
            else:
                ax.plot(
                    d, 
                    #1/result__["thick_empty_pore"]/d,
                    1/result__["thick_empty_pore"],
                    color = "black", 
                    linestyle = "-",
                    label = "$R_{0}$",
                    linewidth = 2,
                    zorder = -1,
                    )
results_ = results_no_fe#.loc[(results_no_fe.mobility_model == model)]
for ax, (chi_PS_, result_) in zip(axs, results_.groupby(by = "chi")):
    #if chi_PC_ not in [-1.0, -1.25, -1.5]: continue
    markers = itertools.cycle(mpl_markers)
    for chi_PC_, result__ in result_.groupby(by = "chi_PC"):
        x = result__["d"].squeeze()
        #y = 1/result__["permeability"]/x
        y = 1/result__["permeability"]

        plot_kwargs = dict(
            label = fr"${chi_PC_:.2f}$",
        )

        ax.plot(
            x, y, 
            **plot_kwargs,
            marker = next(markers),
            mfc = "none",
            ms = 3,
            linewidth = 0.2
            )

for ax in axs:
    ax.grid(linewidth = 0.2)
    #ax.set_ylim(5e-2, 1e4)
    ax.set_ylim(1e-1, 1e3)
    ax.set_xlim(5e-1, 32)
    ax.set_yscale("log")
    ax.set_xscale("log")

    #axs[1,-1].legend()
axs[0].set_ylabel(r"$R \cdot \frac{k_B T}{\eta_0 b}$")
ax.legend(bbox_to_anchor = [1,1])

#axs[1].plot(d, d**3/2)
#plt.tight_layout()
#fig.set_size_inches(7, 7)
fig.set_size_inches(6, 2)
fig.savefig("fig/permeability_on_d.svg")
#fig.savefig("tex/third_report/fig/permeability_on_d_detailed_low_d.svg")
#fig.savefig("tex/third_report/fig/permeability_on_d.svg")
# %%
show_contributions = False
show_CFD = False
show_analytical = True

fig, axs = plt.subplots(
    ncols = 3, 
    #sharey="row", 
    nrows = 1, 
    sharex = False)
first_row_axes = axs
results_ = results.loc[(results.mobility_model == model)]
# results_no_D_ = results_no_D

mpl_markers = ("^",'o', 's', 'D')
for ax, (chi_PC_, result_) in zip(axs, results_.groupby(by = "chi_PC")):
    if chi_PC_ not in [-1.0, -1.25, -1.5]: continue
    ax.set_title(fr"$\chi_{{PC}} = {chi_PC_}$")
    markers = itertools.cycle(mpl_markers)
    for chi_PS_, result__ in result_.groupby(by = "chi"):
        x = result__["d"].squeeze()
        #y = 1/result__["permeability"]/x
        y = 1/result__["permeability"]

        plot_kwargs = dict(
            label = fr"${chi_PS_:.2f}$",
        )

        ax.plot(
            x, y, 
            **plot_kwargs,
            marker = next(markers),
            mfc = "none",
            ms = 3,
            linewidth = 0.2
            )

        if show_CFD:
            R = simulation_results.query(f"(chi_PS=={chi_PS_})&(chi_PC=={chi_PC_})")
            ax.scatter(
                R["d"], 
                #R["R_corrected"]/R["d"],
                R["R_corrected"],
                color = ax.lines[-1].get_color(),
                marker = "s",
                facecolor = "none",
                s=20,
                linewidth = 0.5
                )

    if show_analytical:
        if Haberman_correction_:
            ax.plot(
                d, 
                #1/result__["thick_empty_pore_Haberman"]/d, 
                1/result__["thick_empty_pore_Haberman"],
                color = "black", 
                linestyle = "-",
                label = r"$R^{\ast}_{0}$",
                linewidth = 2,
                zorder = -1,
                )
        else:
            ax.plot(
                d, 
                #1/result__["thick_empty_pore"]/d,
                1/result__["thick_empty_pore"],
                color = "black", 
                linestyle = "-",
                label = "$R_{0}$",
                linewidth = 2,
                zorder = -1,
                )
        # ax.plot(
        #     d, 
        #     1/result__["thin_empty_pore"],
        #     color = "black", 
        #     linestyle = "-",
        #     label = r"$R_{\text{ext}}^{0}$",
        #     linewidth = 1,
        #     zorder = -1,
        # )

    # A = chi_PC_*a1+a0
    # phi_av = 0.4
    # d_star = 2*A/(3*(A**2 - 1))#/(3*chi_PS_-1)
    # ax.axvline(d_star)

results_ = results_no_fe#.loc[(results_no_fe.mobility_model == model)]
for ax, (chi_PC_, result_) in zip(axs, results_.groupby(by = "chi_PC")):
    if chi_PC_ not in [-1.0, -1.25, -1.5]: continue
    ax.set_title(fr"$\chi_{{PC}} = {chi_PC_}$")
    markers = itertools.cycle(mpl_markers)
    for chi_PS_, result__ in result_.groupby(by = "chi"):
        x = result__["d"].squeeze()
        #y = 1/result__["permeability"]/x
        y = 1/result__["permeability"]

        plot_kwargs = dict(
            label = fr"${chi_PS_:.2f}$",
        )

        ax.plot(
            x, y, 
            **plot_kwargs,
            marker = next(markers),
            mfc = "none",
            ms = 3,
            linewidth = 0.2
            )

for ax in axs:
    ax.grid(linewidth = 0.2)
    ax.set_ylim(1e-1, 1e3)
    ax.set_xlim(5e-1, 32)
    ax.set_yscale("log")
    ax.set_xscale("log")

#axs[1].plot(d, d**3/2)
axs[0].set_ylabel(r"$R \cdot \frac{k_B T}{\eta_0 b}$")
ax.legend(bbox_to_anchor = [1,1])
fig.set_size_inches(6, 2)
fig.savefig("fig/permeability_on_d.svg")
# %%
