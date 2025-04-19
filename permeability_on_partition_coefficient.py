# %%
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rc

# rc('text',usetex=True)
# rc('text.latex', preamble=r'\usepackage{color}')
style.use('tableau-colorblind10')


from calculate_fields_in_pore import *
import scf_pb

a0 = 0.70585835
a1 = -0.31406453
wall_thickness=52
pore_radius=26
sigma = 0.02

def free_energy_gel(phi, chi_PS, chi_PC, d):
    V = volume(d)
    S = surface(d)
    Pi_ = Pi(phi, chi_PS, trunc=False)
    gamma_ = gamma(chi_PS, chi_PC, phi, a0, a1)
    free_energy = Pi_*V + gamma_*S
    #print(Pi_)
    # free_energy = gamma_*S
    return free_energy

def PC_gel(phi, chi_PS, chi_PC, d):
    FE = free_energy_gel(phi, chi_PS, chi_PC, d)
    PC = np.exp(-FE)
    return PC

def k_from_normalized_R(
        R_normalized,
        NPC_per_nucleus,
        V_nucleus,          #fL
        eta=0.00145,        #Pa*s
        T=293,              #K
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
    
    return k_ #1/s

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

def chi_from_phi_min_spinodal(N, phi_min):
    """
    Given degree of polymerization N and the polymer volume fraction phi_min
    (on the spinodal), return the exact Flory-Huggins chi parameter.
    """
    return 0.5 * (1.0/(N * phi_min) + 1.0/(1.0 - phi_min))

def chi_from_phi_binodal(phi):
    return (-np.log(1-phi) - phi)/(phi**2)

def get_PC_in_gel(phi_gel, chi_PC, d):
    chi_gel=chi_from_phi_binodal(phi_gel)
    return PC_gel(phi_gel, chi_gel, chi_PC, d)

# def get_thin_pore_permeability(d, r_pore = 26):
#     return empty_pore_permeability(1.0,r_pore-d/2, 0)/(3*np.pi*d)

# def get_dome_permeability(d, r_dome = 26*1.1):
#     return np.pi*r_dome/(3*np.pi*d)

def estimate_protein_diameter(MW_kDa, density=1.35):
    NA = 6.022e23
    # Partial specific volume (cm^3/g)
    v_bar = 1/density
    mw_g_per_mol = MW_kDa * 1000.0
    mass_one_molecule = mw_g_per_mol / NA
    volume_cm3 = mass_one_molecule * v_bar
    volume_nm3 = volume_cm3 * 1.0e21
    radius_nm = ((3.0 * volume_nm3) / (4.0 * np.pi)) ** (1.0 / 3.0)
    diameter_nm = 2.0 * radius_nm
    return diameter_nm

def protein_mass_fraction(
    conc_mg_per_ml: float,
    protein_density: float = 1.35,
    water_density: float = 1.0
) -> float:
    mass_protein_g = conc_mg_per_ml / 1000.0  # g of protein in 1 mL of final solution
    volume_protein_ml = mass_protein_g / protein_density  # (g) / (g/mL) = mL
    volume_water_ml = 1.0 - volume_protein_ml  # 1 mL total solution minus protein volume
    mass_water_g = volume_water_ml * water_density
    mass_total_g = mass_protein_g + mass_water_g
    return mass_protein_g / mass_total_g if mass_total_g > 0 else float('nan')
#%%
experimental_data = pd.read_csv("perm_rates_experimental_data.csv")
globular_protein_density = 1.35#g/ml
Kuhn_segment = 0.76
experimental_data["d"] = experimental_data.apply(lambda _:estimate_protein_diameter(_.MW, globular_protein_density), axis = 1)
experimental_data["d_a"] = experimental_data["d"]/Kuhn_segment
mCherry_influx_rate = 5.7*1e-4
experimental_data["influx_rate"] = experimental_data["Passage_Rate"]*mCherry_influx_rate
#%%
d = [6]
#d = [6,4]
chi_PS = [0.6]
chi_PC = np.round(np.arange(-3, 1.0, 0.05),3)

reference_chi_PC = 0.0
reference_d = 6
protein_density = 1.25#g/ml
#https://www.embopress.org/doi/full/10.1093/emboj/20.6.1320
NPC_per_nucleus = 2770
nucleus_volume = 1130#fL
eta = 0.00145

phi_gel_min = 0.2
phi_gel_max = 0.4
phi_gel_mean = (phi_gel_max+phi_gel_min)/2
show_phi_gel_variation = False
#%%
# model, mobility_model_kwargs = "none", {}
# model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
model, mobility_model_kwargs = "Rubinstein", {"prefactor":30}
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
        convolve_mode="valid",
        mobility_correction= "vol_average",
        mobility_model = model,
        mobility_model_kwargs = mobility_model_kwargs,
        integration="cylindrical_caps",
        cutoff_phi=1e-2,
        Haberman_correction=Haberman_correction_,
        stickiness=True,
        #stickiness_model_kwargs={}
        )
    results.append(result)

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
        cutoff_phi=1e-2,
        Haberman_correction=Haberman_correction_,
        stickiness=False,
        #stickiness_model_kwargs={}
        )
    results.append(result)
results = pd.DataFrame(results)
#%%
#%%
nups = ["Mac98A","Nup116"]
experimental_data_mono=experimental_data.loc[experimental_data.Oligomer == 1]
experimental_data_oligo=experimental_data.loc[experimental_data.Oligomer != 1]
################################################################################
################################################################################
fig, axs = plt.subplots(ncols = len(chi_PS), dpi = 600, sharey=True, sharex= True)
def mark_probe(protein_name, color):
    markers = itertools.cycle(mpl_markers)
    first_it =True
    for nup in nups:
        y = experimental_data_mono.loc[experimental_data_mono.Protein == protein_name,"influx_rate"]
        x = experimental_data_mono.loc[experimental_data_mono.Protein == protein_name,nup]
        s = experimental_data_mono.loc[experimental_data_mono.Protein == protein_name,"d"]
        if first_it: 
            label = protein_name
            first_it = False
        else:
            label = None
        ax.scatter(x,y, color = color, marker = next(markers), s = s*3, label=label)
if len(chi_PS) == 1:
    axs_ = [axs]
else:
    axs_ = axs
results_ = results.loc[(results.mobility_model == model)].sort_values(by = "chi_PC")
ymax = 2e1

for ax, (chi_PS_, result_) in zip(axs_, results_.groupby(by = "chi")):
    y = experimental_data_mono["influx_rate"]
    mpl_markers = ('*', 'D')
    markers = itertools.cycle(mpl_markers) 
    for nup in nups:
        x = experimental_data_mono[nup]
        #s = experimental_data_mono["d"]
        ax.scatter(
            x,y, 
        marker = next(markers), 
        #s=s*3,
        s=15,
        label = nup, 
        color = "black", 
        #fc = "none"
        )
    for nup in nups:
        x = experimental_data_oligo[nup]
        y = experimental_data_oligo["influx_rate"]
        #s = experimental_data_mono["d"]
        ax.scatter(
            x,y, 
        marker = next(markers), 
        #s=s*3,
        s=15,
        #label = nup, 
        color = "k", 
        #fc = "none"
        )
    


    mark_probe("mCherry", "red")
    mark_probe("EGFP", "green")
    #mark_probe("yNTF2 (dimer)", "magenta")
    #mark_probe("rNTF2 (dimer)", "magenta")

    for (d_, stickiness_), result__ in result_.groupby(by = ["d", "stickiness"]):
        permeability_key = "permeability"
        x = get_PC_in_gel(phi_gel_mean, result__["chi_PC"], d_)
        y = result__.apply(lambda _: k_from_normalized_R(
            1/_["permeability"], 
            NPC_per_nucleus, 
            nucleus_volume,
            eta=eta
            ), axis = 1)
        ax.plot(
            x, y, 
            label = fr"$d = {d_}({d_*Kuhn_segment:.1f}"+r"\text{nm})$",
            alpha = 1,
            #color = "tab:blue"
            #marker = next(markers),
            #markevery = 0.5,
            #markersize = 4,
        )
        color_ = ax.lines[-1].get_color()

        if d_==reference_d:
            if show_phi_gel_variation:
                for phi_gel_ in np.linspace(phi_gel_min, phi_gel_max,50):
                    x = get_PC_in_gel(phi_gel_, result__["chi_PC"], d_)
                    ax.plot(
                        x, y, 
                        alpha = 0.1,
                        color = color_
                    )
            # for phi_gel_ in np.linspace(phi_gel_min, phi_gel_max,50):
            #     chi_gel_eq = chi_from_phi_binodal(phi_gel_)
            #     for chi_gel_ in np.linspace(chi_gel_eq-0.05, chi_gel_eq+0.05, 20):
            #         x = PC_gel(phi_gel_,chi_gel_, result__["chi_PC"], d_)
            #         ax.plot(
            #             x, y, 
            #             alpha = 0.1,
            #             color = color_
            #         )
            

        k_empty_pore = get_k_empty_pore(
            
            pore_radius*Kuhn_segment,
            wall_thickness*Kuhn_segment,
            #0,
            d_*Kuhn_segment,
            NPC_per_nucleus,
            nucleus_volume,
            eta=eta,
            #Haberman_correction=True
            )
        ax.axhline(
            k_empty_pore, 
            color =  "black", 
            linestyle = "-", 
            label = "empty pore"
            )


        ax.plot(x.iloc[::10], [ymax]*len(x.to_numpy()[::10]), marker = "|", linewidth = 0, color = "black")
        for xx, ss in zip(x.iloc[::10].squeeze(), result__["chi_PC"].iloc[::10].squeeze()):
            if ss<=-2.0: continue
            if ss>0.0: continue
            ax.text(xx*0.4, ymax*1.1, s = f"{ss:.1f}")



    # Loop over each row in the DataFrame to add text

    # for i, row in experimental_data.iterrows():
    #     y = row["influx_rate"]
    #     if y<2e-4:continue
    #     #for nup in nups:
    #     for nup in ["Nup116"]:
    #         text_key = "Protein"
    #         x = row[nup]
    #         #s = f"{row[text_key]:.1f}"
    #         #s = row[text_key]
    #         s = row.name
    #         ax.text(
    #             x*2.0,
    #             y*0.9,     # small offset in y
    #             s,
    #             ha='center',
    #             va='bottom',
    #             fontsize=5
    #         )

    ax.set_title(r"$\chi_{\text{PC}}$", pad = 18)
    ax.set_ylim(2e-4, ymax)
    ax.set_xlim(1e-2,1e4)
    ax.set_xlabel(r"$c_{\text{eq}}/c_0$")
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.grid()

    #ax.axvline(1, color = "red", linewidth = 0.5)


axs_[0].set_ylabel(r"$k$, $[s^{-1}]$")

#axs_[-1].scatter([],[], marker = "*", color = "grey", label = "empty pore")
axs_[-1].legend( 
    bbox_to_anchor = [1.05, 0.1],
    loc = "lower left"
    )

fig.text(1,0.1, r"$\chi_{\text{PS}}^{\text{pore}} = "+f"{chi_PS_}$")
fig.text(1,0.0, r"$\chi_{\text{PS}}^{\text{gel}} = \{\Pi(\phi_{\text{gel}}) = 0\}$")

#plt.tight_layout()
#fig.set_size_inches(2.5*len(axs_)+0.5, 3)
#plt.tight_layout()
fig.set_size_inches(2.5, 2.5)
fig.savefig("fig/experimental_permeability_on_partitioning.svg")
#%%