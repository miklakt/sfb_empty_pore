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
d = np.arange(2, 32, 2)
#d =[8 ,10, 12 ,]
chi_PS = [0.4, 0.5, 0.6]
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
        truncate_pressure=True,
        method= "convolve",
        convolve_mode="same",
        mobility_correction= "vol_average",
        mobility_model = model,
        mobility_model_kwargs = mobility_model_kwargs,
        integration="cylindrical_caps",
        Haberman_correction=Haberman_correction_,
        stickiness=False,
        )
        
    #result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
    results.append(result)
results = pd.DataFrame(results)

#%%
show_contributions = False
show_CFD = False
show_analytical = True
if show_contributions:
    fig, axs = plt.subplots(ncols = len(chi_PS), sharey="row", nrows = 3, sharex = True)
    first_row_axes = axs[0, :]
else:
    fig, axs = plt.subplots(ncols = len(chi_PS), sharey="row", nrows = 1, sharex = True)
    first_row_axes = axs
results_ = results.loc[(results.mobility_model == model)]

for ax, (chi_PS_, result_) in zip(first_row_axes, results_.groupby(by = "chi")):
    markers = itertools.cycle(mpl_markers)
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
                label = "$R_{empty}$",
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
                label = "$R_{empty}$",
                linewidth = 2,
                zorder = -1,
                )

        # ax.plot(
        #     d, 
        #     1/(2*(pore_radius))*3*np.pi*d/d,
        #     color = "black", 
        #     linestyle = "-",
        #     linewidth = 0.5,
        #     #label = "$P_{thin}$",
        #     zorder = -1,
        #     )

        # ax.plot(
        #     d, 
        #     1/(2*np.pi*(pore_radius))*3*np.pi*d/d,
        #     color = "black", 
        #     linestyle = "-",
        #     #label = "$P_{thin}$",
        #     linewidth = 0.5,
        #     zorder = -1,
        #     )
        
        # ax.plot(
        #     d, 
        #     1/(2*np.pi*(pore_radius+5+d/2))*3*np.pi*d/d,
        #     color = "black", 
        #     linestyle = "-",
        #     #label = "$P_{thin}$",
        #     linewidth = 0.5,
        #     zorder = -1,
        #     )
        
        # ax.plot(
        #     d, 
        #     1/(2*np.pi*(pore_radius+d/2))*3*np.pi*d,
        #     color = "black", 
        #     linestyle = "-",
        #     #label = "$P_{thin}$",
        #     linewidth = 0.5,
        #     zorder = -1,
        #     )


        # R_cylinder = wall_thickness/(np.pi*(pore_radius-d/2)**2)/result__["einstein_factor"]
        # ax.plot(
        #     d, 
        #     R_cylinder, 
        #     color = "black", 
        #     linestyle = "-.",
        #     label = "$P_{thin}$",
        #     zorder = -1,
        #     )

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

    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    #ax.set_ylim(5e-2, 1e4)
    ax.set_ylim(5e-1, 1e4)
    #ax.set_xlim(4)
    ax.set_yscale("log")
    ax.set_xscale("log")

if show_contributions:
    for ax, (chi_PS_, result_) in zip(axs[1,:], results_.groupby(by = "chi")):
        markers = itertools.cycle(mpl_markers)
        for chi_PC_, result__ in result_.groupby(by = "chi_PC"):
            x = result__["d"].squeeze()
            R0 = 1/result__["thin_empty_pore"]
            y = (result__["R_left"] + result__["R_right"])/R0
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
                **plot_kwargs
                )
        
        ax.axhline(1, color = "black", linestyle = ":", linewidth = 2, zorder = -1)

        #ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
        ax.set_ylim(5e-2, 1e3)
        #ax.set_xlabel("d")
        ax.set_yscale("log")
        ax.set_xscale("log")

    # for ax, (chi_PS_, result_) in zip(axs[2,:], results_.groupby(by = "chi")):
    #     markers = itertools.cycle(mpl_markers)
    #     for chi_PC_, result__ in result_.groupby(by = "chi_PC"):
    #         x = result__["d"].squeeze()
    #         R0 = (wall_thickness+d)/(np.pi*(pore_radius-d/2)**2)/result__["einstein_factor"]
    #         y = (result__["R_pore"])/R0
    #         if chi_PC_ in chi_PC_color:
    #             plot_kwargs = dict(
    #                 label = fr"$\chi_{{PC}} = {chi_PC_}$",
    #                 #marker = next(markers),
    #                 #markevery = 0.5,
    #                 #markersize = 4,
    #             )
    #         else:
    #             plot_kwargs = dict(
    #                 linewidth = 0.1,
    #                 color ="black"
    #             )
    #         ax.plot(
    #             x, y, 
    #             **plot_kwargs
    #             )

    #     ax.axhline(1, color = "black", linestyle = "-.", linewidth = 2, zorder = -1)

    #     ax.set_ylim(1e-3, 1e3)
    #     ax.set_xlabel("d")
    #     ax.set_yscale("log")
    #     ax.set_xscale("log")

    axs[1,0].set_ylabel(r"$R_{conv}/R_{thin}(d)$")
    axs[2,0].set_ylabel(r"$\frac{R_{cylinder}}{\pi(r-d/2)^2} \frac{k_B T}{3 \pi \eta_0 d}$")
    # axs[1,-1].legend( 
    #     bbox_to_anchor = [1.0, -0.05],
    #     loc = "lower left"
    #     )
    axs[1,-1].plot([],[], color = "black", linestyle = ":", linewidth = 2,

                    label = "$R_{conv}$"
                    )

    axs[1,-1].plot([],[], color = "black", linestyle = "-.", linewidth = 2,
                    label = "$R_{channel}$"
                    )

    axs[1,-1].plot([],[], color = "black", linestyle = "-", linewidth = 2,
                    label = "$R_{empty}$"
                    )

    #axs[1,-1].legend()
axs[0].set_ylabel(r"$R \cdot \frac{k_B T}{\eta_0 b}$")
ax.legend(bbox_to_anchor = [1,1])
#plt.tight_layout()
#fig.set_size_inches(7, 7)
fig.set_size_inches(6, 2)
fig.savefig("fig/permeability_on_d.svg")
#fig.savefig("tex/third_report/fig/permeability_on_d_detailed_low_d.svg")
#fig.savefig("tex/third_report/fig/permeability_on_d.svg")
# %%
