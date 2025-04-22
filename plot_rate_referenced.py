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
#%%
simulation_results = pd.read_csv("numeric_simulation_results.csv")

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
d = np.arange(2, 20, 2)
#d =[8 ,10, 12 ,]
chi_PS = [0.6]
#chi_PC = [-2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1, -0.75]
chi_PC_color = [0, -0.5]
chi_PC = chi_PC_color

# model, mobility_model_kwargs = "none", {}
#model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
# model = "Fox-Flory", dict(N = 300)
model, mobility_model_kwargs = "Rubinstein", {"prefactor":30}
#model, mobility_model_kwargs = "Rubinstein", {"prefactor":1, "nu":0.5}
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
        Haberman_correction=Haberman_correction_
        )
        
    #result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
    results.append(result)
results = pd.DataFrame(results)
#%%
experimental_data=pd.DataFrame(
    {
    "Probe": [
        "Fluorescein-Cys",
        "11 aa peptide",
        "Insulin",
        "Aprotinin",
        "Profilin",
        "Ubiquitin",
        "z-domain",
        "Thioredoxin",
        "Lactalbumin",
        "GFP",
        "PBP",
        "MBP",
        "mCherry"#
    ],
    "stokes_r_nm": [
        0.67,
        0.91,
        1.19,
        1.48,
        1.65,
        1.69,
        1.71,
        1.97,
        2.07,
        2.42,
        2.75,
        2.85,
        2.45,#
    ],
    "Influx_rate": [
        0.940,
        0.53,
        0.24,
        0.086,
        0.0548,
        0.0356,
        0.0401,
        0.0203,
        0.0144,
        0.00205,
        0.00026,
        0.00022,
        5.7*1e-4#
    ],
    "qi": [
        46,
        26,
        11.8,
        4.25,
        2.70,
        1.75,
        1.98,
        1.00,
        0.707,
        0.1010,
        0.0126,
        0.0109,
        0.028#
        
    ],
    "qi_std": [
        9.2,
        3.9,
        1.31,
        0.58,
        0.40,
        0.28,
        0.28,
        np.nan,
        0.012,
        0.0140,
        0.0071,
        0.0028,
        np.nan#
    ],
}
)

reference_particle_radius = 1.97#nm

Kuhn_segment = 0.76
experimental_data["d"] = experimental_data["stokes_r_nm"]*2#/Kuhn_segment*2
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
def k_from_normalized_R(
        R_normalized,
        NPC_per_nucleus,
        V_nucleus,          #fL
        eta=0.00145,        #Pa*s
        T=293,              #K
        segment_length = 0.76 #nm
        ):
    k_B = 1.380649*1e-23   #J/K
    V_nucleus_ = V_nucleus*1e-18   #m^3
    R_ = R_normalized*eta/(k_B * T) #s/m^3
    k_ = NPC_per_nucleus/V_nucleus_/R_
    return k_

def get_k_empty_pore(
        r_p,                #nm
        L,                  #nm
        d,                  #nm
        NPC_per_nucleus,
        V_nucleus,          #fL
        eta=0.00145,           #Pa*s
        T=293,              #K
        Haberman_correction = False
        ):
    k_B = 1.380649*1e-23   #J/K

    r_p_ = (r_p-d/2)*1e-9            #m^3
    L_ = (L+d)*1e-9                #m^3
    d_ = d*1e-9                      #m^3
    eta_ = eta                       #Pa*s
    V_nucleus_ = V_nucleus*1e-18     #m^3

    D_0_ = k_B * T / (3 * np.pi * eta_ * d_)  #m^2/s
    if Haberman_correction:
        from calculate_fields_in_pore import Haberman_correction_approximant
        K = float(Haberman_correction_approximant(d, r_p))
    else:
        K = 1.0
    R_0_ = K*L_ / (D_0_ * np.pi * r_p_**2) + 1 / (2 * D_0_ * r_p_) #s/m^3
    k_ = NPC_per_nucleus/V_nucleus_/R_0_       #1/s
    
    return k_
#%%
# fig, axs = plt.subplots(ncols = len(chi_PS), sharey="row", nrows = 1, sharex = True)
# if len(chi_PS) == 1:
#     axs_ = [axs]
# else:
#     axs_ = axs

# #reference = results.loc[(results.d==6)]



# for ax, (chi_PS_, result_) in zip(axs_, results.groupby(by = "chi")):
#     markers = itertools.cycle(mpl_markers)
#     for chi_PC_, result__ in result_.groupby(by = "chi_PC"):
        
#         #reference_ = reference.loc[(reference.chi == chi_PS_)&(reference.chi_PC == chi_PC_), "permeability"].squeeze()

#         x = result__["d"].squeeze()*Kuhn_segment
#         y = result__["permeability"]

#         interp_func = create_interp_func(x,np.log(y))

#         reference_permeability = np.exp(interp_func(reference_particle_radius*2))
#         y = result__["permeability"]/reference_permeability

#         if chi_PC_ in chi_PC_color:
#             plot_kwargs = dict(
#                 label = fr"$\chi_{{\text{{PC}}}} = {chi_PC_}$",
#                 #label = fr"${chi_PC_:.2f}$",
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
#             **plot_kwargs,
#             marker = next(markers),
#             #mfc = "none",
#             ms = 3,
#             linewidth = 0.5
#             )

#         ax.scatter(
#             experimental_data["d"], 
#             experimental_data["qi"], 
#             color = "black", 
#             linewidth = 0.1, 
#             marker = "*", 
#             )

#     ax.scatter(
#         [], 
#         [], 
#         color = "black", 
#         linewidth = 0.1, 
#         marker = "*", 
#         label ="experimental"
#     )

#     reference_probe = "Thioredoxin"
#     for idx, row in experimental_data.iterrows():
#         if row["Probe"] == reference_probe:
#             color = "Red"
#         else:
#             color = None
#         ax.text(
#             x = row["d"],
#             y = row["qi"],
#             s = row["Probe"],
#             rotation = 90,
#             va = "top",
#             ha = "center",
#             color = color
#         )
    
#     ref_record = experimental_data.loc[experimental_data["Probe"] == reference_probe]
#     x = ref_record["d"].squeeze()
#     y = ref_record["qi"].squeeze()
#     ax.scatter(
#         [x], 
#         [y], 
#         color = "red", 
#         linewidth = 0.1, 
#         marker = "*", 
#         #label =reference_probe
#     )

#     reference_probe = "GFP"
#     ref_record = experimental_data.loc[experimental_data["Probe"] == reference_probe]
#     x = ref_record["d"].squeeze()
#     y = ref_record["qi"].squeeze()
#     ax.scatter(
#         [x], 
#         [y], 
#         color = "green", 
#         linewidth = 0.1, 
#         marker = "*", 
#         #label =reference_probe
#     )


#     ax.set_yscale("log")
#     ax.set_title(r"$\chi_{\text{PS}}="+f"{chi_PS_}$")
#     ax.set_xlabel("d, nm")
#     ax.set_ylim(1e-2,1e2)
#     ax.set_xlim(1.2,6.2)

# axs_[0].set_ylabel(rf"$R(d = {reference_particle_radius*2}"+r"\text{nm}) / R$")
# ax.legend(
#     #title = r"$\chi_{\text{PC}}$"
#     )
# #fig.set_size_inches(2.5*len(chi_PS)+1, 3)
# plt.tight_layout()
# fig.set_size_inches(2.5, 2.5)
# #fig.savefig("fig/experimental_inert.svg")
# %%
fig, axs = plt.subplots(ncols = len(chi_PS), sharey="row", nrows = 1, sharex = True)
if len(chi_PS) == 1:
    axs_ = [axs]
else:
    axs_ = axs

#reference = results.loc[(results.d==6)]



for ax, (chi_PS_, result_) in zip(axs_, results.groupby(by = "chi")):
    markers = itertools.cycle(mpl_markers)
    for chi_PC_, result__ in result_.groupby(by = "chi_PC"):
        #https://www.embopress.org/doi/full/10.1093/emboj/20.6.1320
        NPC_per_nucleus = 2770
        nucleus_volume = 1130#fL
        eta = 0.00145

        x = result__["d"].squeeze()*Kuhn_segment

        y = result__.apply(lambda _: k_from_normalized_R(1/_["permeability"], 
                        NPC_per_nucleus, 
                        nucleus_volume,
                        eta = eta,
                        ), axis = 1)

        #y=y/10
        x2 = np.linspace(1,6)
        y2 = get_k_empty_pore(
                    pore_radius*Kuhn_segment,
                    wall_thickness*Kuhn_segment,
                    x2,
                    NPC_per_nucleus,
                    nucleus_volume,
                    eta=eta,
                    )
        
        ax.plot(
            x2, y2, 
            label = "empty",
            #marker = next(markers),
            #mfc = "none",
            #ms = 3,
            linewidth = 1,
            linestyle = "--",
            color = "black"
            )
        
        y2 = [get_k_empty_pore(
                    pore_radius*Kuhn_segment,
                    wall_thickness*Kuhn_segment,
                    x2_,
                    NPC_per_nucleus,
                    nucleus_volume,
                    eta=eta,
                    Haberman_correction=True
                    ) for x2_ in x2]
        
        ax.plot(
            x2, y2, 
            label = "empty",
            #marker = next(markers),
            #mfc = "none",
            #ms = 3,
            linewidth = 0.5,
            linestyle = "--",
            color = "black"
            )
        

        if chi_PC_ in chi_PC_color:
            plot_kwargs = dict(
                label = fr"$\chi_{{\text{{PC}}}} = {chi_PC_}$",
                #label = fr"${chi_PC_:.2f}$",
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
            linewidth = 0.5,
            color = "tab:blue"
            )

        ax.scatter(
            experimental_data["d"], 
            experimental_data["Influx_rate"], 
            color = "black", 
            linewidth = 0.1, 
            marker = "*", 
            )

    ax.scatter(
        [], 
        [], 
        color = "black", 
        linewidth = 0.1, 
        marker = "*", 
        label ="experimental"
    )

    reference_probe = "Thioredoxin"
    # for idx, row in experimental_data.iterrows():
    #     if row["Probe"] == reference_probe:
    #         color = "Red"
    #     else:
    #         color = None
    #     ax.text(
    #         x = row["d"],
    #         y = row["Influx_rate"],
    #         s = row["Probe"],
    #         rotation = 90,
    #         va = "top",
    #         ha = "center",
    #         color = color
    #     )
    
    ref_record = experimental_data.loc[experimental_data["Probe"] == reference_probe]
    x = ref_record["d"].squeeze()
    y = ref_record["Influx_rate"].squeeze()
    ax.scatter(
        [x], 
        [y], 
        color = "red", 
        linewidth = 0.1, 
        marker = "*", 
        #label =reference_probe
    )

    reference_probe = "GFP"
    ref_record = experimental_data.loc[experimental_data["Probe"] == reference_probe]
    x = ref_record["d"].squeeze()
    y = ref_record["Influx_rate"].squeeze()
    ax.scatter(
        [x], 
        [y], 
        color = "green", 
        linewidth = 0.1, 
        marker = "*", 
        #label =reference_probe
    )

    reference_probe = "mCherry"
    ref_record = experimental_data.loc[experimental_data["Probe"] == reference_probe]
    x = ref_record["d"].squeeze()
    y = ref_record["Influx_rate"].squeeze()
    ax.scatter(
        [x], 
        [y], 
        color = "green", 
        linewidth = 0.1, 
        marker = "*", 
        #label =reference_probe
    )


    ax.set_yscale("log")
    ax.set_title(r"$\chi_{\text{PS}}="+f"{chi_PS_}$")
    ax.set_xlabel("d, nm")
    ax.set_ylim(1e-4,5e1)
    ax.set_xlim(1.2,6.2) 

axs_[0].set_ylabel(r"$k \, [s^{-1}]$")
# ax.legend(
#     #title = r"$\chi_{\text{PC}}$"
#     )
#fig.set_size_inches(2.5*len(chi_PS)+1, 3)
plt.tight_layout()
fig.set_size_inches(2.5, 2.5)
#fig.savefig("fig/experimental_inert.svg")
# %%
