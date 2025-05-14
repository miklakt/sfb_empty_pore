#%%
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

a0 = 0.7
a1 = -0.3
wall_thickness=52
pore_radius=26
sigma = 0.02
#%%
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

#%%
#https://doi.org/10.1083/jcb.201601004
experimental_data_2 = pd.DataFrame(
    {
    "Probe":["GFP-HIS", "GFP-1PrA", "GFP-2PrA", "GFP-3PrA","GFP-4PrA", "GFP-6PrA", "GFP-1PrG", "GFP-2PrG"],
    "MM":[ 26.8, 34.2, 40.7,  46.8,  53.6,   66.8,  34.7,  42.3],
    "Rg":[ 2.5,  3.1,  3.3,   3.7,   3.9,    4.3,   3.2,   3.5],
    "tau":[15,   62,   114,   180,   252,    413,   20,    66],
    "tau_err":[1.4, np.nan, np.nan, np.nan, np.nan, 92, np.nan, np.nan],
    "D":[  9.33, 8.42, 7.1,   6.59,  6.17,   5.67, np.nan, np.nan]
    }
)
#http://www.molbiolcell.org/cgi/doi/10.1091/mbc.E14-07-1175
experimental_data_3 = pd.DataFrame(
    {
    "Probe":["MG",   "MGM",   "MGM2",   "MGM4",   "MG2",   "MG3",   "MG4",   "MG5"],
    "MM": [   68,     109,     149,      230,      95,       122,     150,     177],
    "NC" :[np.nan,    0.85,    0.4,      0.21,     np.nan,   0.66,    0.53,    0.33]
    }
)

#https://www.embopress.org/doi/full/10.1093/emboj/20.6.1320
experimental_data_4 = pd.DataFrame({
    "Probe":["Large", "Transportin"],
    "MM":[630, 100],
    "Translocations" : [28, 65]
})
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
        V_cytoplasm = None,
        ):
    k_B = 1.380649*1e-23   #J/K
    V_nucleus_ = V_nucleus*1e-18   #m^3
    R_ = R_normalized*eta/(k_B * T) #s/m^3
    if V_cytoplasm is None:
        k_ = NPC_per_nucleus/R_*(1/V_nucleus_)
    else:
        V_cytoplasm_ = V_cytoplasm*0e-18   #m^3
        k_ = NPC_per_nucleus/R_*(1/V_nucleus_+1/V_cytoplasm_)
    return k_

def R_from_k(
        k,                  #s^-1
        NPC_per_nucleus,
        V_nucleus,          #fL
        eta=0.00145,        #Pa*s
        T=293,              #K
        V_cytoplasm = None,
        normalize =True
    ):
    k_B = 1.380649*1e-23   #J/K
    V_nucleus_ = V_nucleus*1e-18   #m^3
    if V_cytoplasm is None:
        R_ = NPC_per_nucleus/k*(1/V_nucleus_)
    else:
        V_cytoplasm_ = V_cytoplasm*1e-18   #m^3
        R_ = NPC_per_nucleus/k*(1/V_nucleus_+1/V_cytoplasm_)
    if normalize:
        R = R_*k_B*T/eta
    else:
        R = R_
    return R



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
    L_ = (L+d)*1e-9                  #m^3
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

def eta_from_d(
        D, #cm2/s
        d, #nm
        T = 293 #K
        ):#Pa*s
    k_B = 1.380649*1e-23
    d_ = d*1e-9
    D_ = D*1e-4
    eta = k_B*T/(3*np.pi*d_*D_)
    return eta

def tau_from_nc_ratio(conc_ratio, volume_ratio, time):
    r = (conc_ratio*volume_ratio+1)/(1-conc_ratio)
    tau = time/np.log(r)
    return tau

def estimate_protein_diameter(MW_kDa, density=1.2):
    NA = 6.022e23
    # Partial specific volume (cm^3/g)
    v_bar = 1/density
    mw_g_per_mol = MW_kDa * 1000.0
    mass_one_molecule = mw_g_per_mol / NA
    volume_cm3 = mass_one_molecule * v_bar
    volume_nm3 = volume_cm3 * 1.0e21
    radius_nm = ((3.0 * volume_nm3) / (4.0 * np.pi)) ** (1.0 / 3.0)
    diameter_nm = 2.0 * radius_nm
    #diameter_nm = 0.066*(MW_kDa*1000)**(0.37)*2
    return diameter_nm

def estimate_molecular_weight(diameter_nm, density=1.2):
    """
    Estimate the molecular weight (in kDa) of a globular protein given its diameter in nanometers.
    
    Parameters:
        diameter_nm (float): Estimated diameter of the protein in nanometers.
        density (float): Protein density in g/cm^3 (default is 1.1).
    
    Returns:
        float: Estimated molecular weight in kilodaltons (kDa).
    """
    NA = 6.022e23  # Avogadro's number
    v_bar = 1 / density  # cm^3/g
    radius_nm = diameter_nm / 2.0
    volume_nm3 = (4.0 / 3.0) * np.pi * radius_nm**3
    volume_cm3 = volume_nm3 * 1.0e-21
    mass_one_molecule = volume_cm3 / v_bar
    mw_g_per_mol = mass_one_molecule * NA
    mw_kDa = mw_g_per_mol / 1000.0
    return mw_kDa
#%%
experimental_data["d"] = estimate_protein_diameter(experimental_data["MM"])
#%%
#yeast NPC
nucleus_volume_ = 4.8 #fl
cytoplasm_volume_ = 60 #fl
NPC_per_nucleus_ = 161

NPC_per_nucleus = 2770
nucleus_volume = 1130#fL
experimental_data_2["tau_renormalized"] = \
    nucleus_volume/NPC_per_nucleus * \
    (nucleus_volume_+cytoplasm_volume_)/(nucleus_volume_*cytoplasm_volume_) *\
    NPC_per_nucleus_*experimental_data_2["tau"]
#experimental_data_2["eta"] = experimental_data_2.apply(lambda _: eta_from_d(_["D"], _["d"]), axis =1)
experimental_data_2["d"] = estimate_protein_diameter(experimental_data_2["MM"])
#%%
experimental_data_3["tau"] =  experimental_data_3.apply(lambda _: tau_from_nc_ratio(_["NC"], nucleus_volume_/cytoplasm_volume_, 1*60*60), axis =1)
experimental_data_3["d"] = estimate_protein_diameter(experimental_data_3["MM"])

experimental_data_3["tau_renormalized"] = \
    nucleus_volume/NPC_per_nucleus *\
    (nucleus_volume_+cytoplasm_volume_)/(nucleus_volume_*cytoplasm_volume_) *\
    NPC_per_nucleus_*experimental_data_3["tau"] 
#%%
plt.scatter(experimental_data_2["MM"], experimental_data_2["tau"])
plt.scatter(experimental_data_3["MM"], experimental_data_3["tau"])
plt.xscale("log")
plt.yscale("log")
x = np.arange(20, 300)
y = x**3.2/1700
plt.plot(x,y)
#%%
normalize = False
experimental_data["R"] = experimental_data.apply(lambda _: R_from_k(_["Influx_rate"], NPC_per_nucleus, nucleus_volume, normalize = normalize), axis = 1)
experimental_data_2["R"] = experimental_data_2.apply(lambda _: R_from_k(_["tau"]**(-1), NPC_per_nucleus_, nucleus_volume_, V_cytoplasm = cytoplasm_volume_, normalize = normalize), axis = 1)
experimental_data_3["R"] = experimental_data_3.apply(lambda _: R_from_k(_["tau"]**(-1), NPC_per_nucleus_, nucleus_volume_, V_cytoplasm = cytoplasm_volume_, normalize = normalize), axis = 1)
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
        x2 = np.linspace(1,12)
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

        #Rigid pore
        # x3 = np.linspace(1,9.5)
        # y3 = [get_k_empty_pore(
        #             5,
        #             wall_thickness*Kuhn_segment,
        #             x3_,
        #             NPC_per_nucleus,
        #             nucleus_volume,
        #             eta=eta,
        #             Haberman_correction=True
        #             ) for x3_ in x3]

        # ax.plot(
        #     x3, y3, 
        #     label = "empty",
        #     #marker = next(markers),
        #     #mfc = "none",
        #     #ms = 3,
        #     linewidth = 1,
        #     linestyle = "--",
        #     color = "grey"
        #     )
        
        # y2 = [expression(x2_,
        #             ) for x2_ in x2]
        
        # ax.plot(
        #     x2, y2, 
        #     label = "empty",
        #     #marker = next(markers),
        #     #mfc = "none",
        #     #ms = 3,
        #     linewidth = 0.5,
        #     linestyle = "--",
        #     color = "red"
        #     )
        

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
            experimental_data_2["d"], 
            experimental_data_2["tau_renormalized"]**(-1), 
            color = "black", 
            linewidth = 0.1, 
            marker = "s", 
            )

        ax.scatter(
            experimental_data_3["d"], 
            experimental_data_3["tau_renormalized"]**(-1), 
            color = "black", 
            linewidth = 0.1, 
            marker = "^", 
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
    
    # ref_record = experimental_data.loc[experimental_data["Probe"] == reference_probe]
    # x = ref_record["d"].squeeze()
    # y = ref_record["Influx_rate"].squeeze()
    # ax.scatter(
    #     [x], 
    #     [y], 
    #     color = "red", 
    #     linewidth = 0.1, 
    #     marker = "*", 
    #     #label =reference_probe
    # )

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

    reference_probe = "GFP-HIS"
    ref_record = experimental_data_2.loc[experimental_data_2["Probe"] == reference_probe]
    x = ref_record["d"].squeeze()
    y = ref_record["tau_renormalized"].squeeze()**(-1)
    ax.scatter(
        [x], 
        [y], 
        color = "green", 
        linewidth = 0.1, 
        marker = "s", 
        #label =reference_probe
    )

    reference_probe = "mCherry"
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

    reference_probe = "MBP"
    ref_record = experimental_data.loc[experimental_data["Probe"] == reference_probe]
    x = ref_record["d"].squeeze()
    y = ref_record["Influx_rate"].squeeze()
    ax.scatter(
        [x], 
        [y], 
        color = "magenta", 
        linewidth = 0.1, 
        marker = "*", 
        #label =reference_probe
    )


    ax.set_yscale("log")
    ax.set_title(r"$\chi_{\text{PS}}="+f"{chi_PS_}$")
    ax.set_xlabel("d, nm")
    ax.set_ylim(1e-6,5e1)
    ax.set_xlim(1.2,12) 

    ax.grid()

axs_[0].set_ylabel(r"$k \, [s^{-1}]$")
# ax.legend(
#     #title = r"$\chi_{\text{PC}}$"
#     )
#fig.set_size_inches(2.5*len(chi_PS)+1, 3)
plt.tight_layout()
fig.set_size_inches(3, 3)
#fig.savefig("fig/experimental_inert.svg")
# %%
fig, axs = plt.subplots(ncols = len(chi_PS), sharey="row", nrows = 1, sharex = True)
if len(chi_PS) == 1:
    axs_ = [axs]
else:
    axs_ = axs

#reference = results.loc[(results.d==6)]

NA = 6.02214076*1e23

for ax, (chi_PS_, result_) in zip(axs_, results.groupby(by = "chi")):
    markers = itertools.cycle(mpl_markers)
    for chi_PC_, result__ in result_.groupby(by = "chi_PC"):
        #https://www.embopress.org/doi/full/10.1093/emboj/20.6.1320
        NPC_per_nucleus = 2770
        nucleus_volume = 1130#fL
        eta = 0.00145

        x = estimate_molecular_weight(result__["d"].squeeze()*Kuhn_segment)

        # y = result__.apply(lambda _: k_from_normalized_R(1/_["permeability"], 
        #                 NPC_per_nucleus, 
        #                 nucleus_volume,
        #                 eta = eta,
        #                 ), axis = 1)
        
        k_B = 1.380649*1e-23
        T=293
        y = result__["permeability"]**(-1) / (k_B*T/eta) / NA
        y = y**-1
        # #y=y/10
        # x2 = np.linspace(1,12)
        # y2 = get_k_empty_pore(
        #             pore_radius*Kuhn_segment,
        #             wall_thickness*Kuhn_segment,
        #             x2,
        #             NPC_per_nucleus,
        #             nucleus_volume,
        #             eta=eta,
        #             )
        # x2=estimate_molecular_weight(x2) 
        # ax.plot(
        #     x2, y2, 
        #     label = "empty",
        #     #marker = next(markers),
        #     #mfc = "none",
        #     #ms = 3,
        #     linewidth = 1,
        #     linestyle = "--",
        #     color = "black"
        #     )

       
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
            experimental_data["MM"], 
            experimental_data["R"]**(-1)*NA, 
            color = "black", 
            linewidth = 0.1, 
            marker = "*", 
            )
        
        ax.scatter(
            experimental_data_2["MM"], 
            experimental_data_2["R"]**(-1)*NA, 
            color = "black", 
            linewidth = 0.1, 
            marker = "s", 
            )

        ax.scatter(
            experimental_data_3["MM"], 
            experimental_data_3["R"]**(-1)*NA, 
            color = "black", 
            linewidth = 0.1, 
            marker = "^", 
            )

        # ax.scatter(
        #     experimental_data_4["MM"], 
        #     experimental_data_4["Translocations"], 
        #     color = "red", 
        #     linewidth = 0.1, 
        #     marker = "o", 
        #     )

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
    
    # ref_record = experimental_data.loc[experimental_data["Probe"] == reference_probe]
    # x = ref_record["d"].squeeze()
    # y = ref_record["Influx_rate"].squeeze()
    # ax.scatter(
    #     [x], 
    #     [y], 
    #     color = "red", 
    #     linewidth = 0.1, 
    #     marker = "*", 
    #     #label =reference_probe
    # )

    reference_probe = "GFP"
    ref_record = experimental_data.loc[experimental_data["Probe"] == reference_probe]
    x = ref_record["MM"].squeeze()
    y = ref_record["R"].squeeze()**(-1)*NA
    ax.scatter(
        [x], 
        [y], 
        color = "green", 
        linewidth = 0.1, 
        marker = "*", 
        #label =reference_probe
    )

    reference_probe = "GFP-HIS"
    ref_record = experimental_data_2.loc[experimental_data_2["Probe"] == reference_probe]
    x = ref_record["MM"].squeeze()
    y = ref_record["R"].squeeze()**(-1)*NA
    ax.scatter(
        [x], 
        [y], 
        color = "green", 
        linewidth = 0.1, 
        marker = "s", 
        #label =reference_probe
    )

    reference_probe = "mCherry"
    ref_record = experimental_data.loc[experimental_data["Probe"] == reference_probe]
    x = ref_record["MM"].squeeze()
    y = ref_record["R"].squeeze()**(-1)*NA
    ax.scatter(
        [x], 
        [y], 
        color = "red", 
        linewidth = 0.1, 
        marker = "*", 
        #label =reference_probe
    )

    reference_probe = "MBP"
    ref_record = experimental_data.loc[experimental_data["Probe"] == reference_probe]
    x = ref_record["MM"].squeeze()
    y = ref_record["R"].squeeze()**(-1)*NA
    ax.scatter(
        [x], 
        [y], 
        color = "magenta", 
        linewidth = 0.1, 
        marker = "*", 
        #label =reference_probe
    )


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"$\chi_{\text{PS}}="+f"{chi_PS_}$")
    ax.set_xlabel(r"$\sqrt[3]{\text{MM}}, \sqrt[3]{\text{kDa}}$")
    ax.set_ylim(1e-1,1e6)
    ax.set_xlim(1,300) 

    ax.grid()

#axs_[0].set_ylabel(r"$R\, [\text{s}/\text{m}^3]$")
axs_[0].set_ylabel("Number of translocation through NPC\n" +r"at $\Delta c = 1 \mu\text{M}$")
# ax.legend(
#     #title = r"$\chi_{\text{PC}}$"
#     )
#fig.set_size_inches(2.5*len(chi_PS)+1, 3)
plt.tight_layout()
fig.set_size_inches(3, 3)
#fig.savefig("fig/experimental_inert.svg")
#%%