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
    return free_energy

def PC_gel(phi, chi_PS, chi_PC, d):
    FE = free_energy_gel(phi, chi_PS, chi_PC, d)
    PC = np.exp(-FE)
    return PC
#%%
#d = np.arange(6, 24, 2)
d = [6]
chi_PS = [0.5, 0.6]
chi_PC = np.round(np.arange(-3, 0.2, 0.05),3)

# model, mobility_model_kwargs = "none", {}
# model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
# model = "Fox-Flory", dict(N = 300)
model, mobility_model_kwargs = "Rubinstein", {"prefactor":1}
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
        convolve_mode="valid",
        mobility_correction= "vol_average",
        mobility_model = model,
        mobility_model_kwargs = mobility_model_kwargs,
        integration="cylindrical_caps",
        cutoff_phi=1e-2,
        Haberman_correction=Haberman_correction_
        )
    
    #result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
    
    results.append(result)
results = pd.DataFrame(results)
#%%
experimental_data = pd.read_csv("perm_rates_experimental_data.csv")
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

def chi_from_phi_min_spinodal(N, phi_min):
    """
    Given degree of polymerization N and the polymer volume fraction phi_min
    (on the spinodal), return the exact Flory-Huggins chi parameter.
    """
    return 0.5 * (1.0/(N * phi_min) + 1.0/(1.0 - phi_min))

def estimate_protein_diameter(MW_kDa, density=1.35):
    NA = 6.022e23
    MW_g_per_mol = MW_kDa * 1000.0
    mass_per_molecule_g = MW_g_per_mol / NA
    volume_cm3 = mass_per_molecule_g / density
    diameter_nm = ((3.0 * volume_cm3) / (2.0 * np.pi)) ** (1.0 / 3.0) * 1e7
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
reference_chi_PC = 0.0
#reference_d = 6
Kuhn_segment = 0.76
#phi_gel = 0.2
#chi_gel = 0.65

protein_density = 1.25#g/ml
phi_gel = {
    "Mac98A":protein_mass_fraction(
                conc_mg_per_ml=175, 
                protein_density=protein_density
                ),
    "Nup116":protein_mass_fraction(
                conc_mg_per_ml=200, 
                protein_density=protein_density
                ),
    "Nsp1": protein_mass_fraction(
                conc_mg_per_ml=200, 
                protein_density=protein_density
                )
}

# phi_gel = protein_mass_fraction(
#     conc_mg_per_ml=250, 
#     protein_density=1.3
#     ) #0.24

chi_gel = {key:chi_from_phi_min_spinodal(N=300, phi_min=phi_gel[key]) for key in phi_gel}
# chi_gel = chi_from_phi_min_spinodal(
#     N=300, phi_min=phi_gel
#     )#0.664

#%%
phi_gel=0.2
chi_gel=0.65
results["PC_gel"]=results.apply(lambda _:PC_gel(phi_gel, chi_gel, _["chi_PC"], _["d"]), axis = 1)

def get_permeability_different_bc(boundary_z, permeability):
    R_reservoir = (-np.log(pore_radius/3 + boundary_z) + np.log(pore_radius + boundary_z))/(2*np.pi*pore_radius)
    return (permeability**(-1)-R_reservoir)**(-1)

def get_permeability_longer_pore(add_length, permeability_z, permeability):
    R_added  = (permeability_z[int(len(permeability_z)/2)])**(-1)*add_length
    return (permeability**(-1)+R_added)**(-1)


# add_length=150
# results["permeability_longer"]=results.apply(lambda _:get_permeability_longer_pore(add_length, _.permeability_z, _.permeability), axis = 1)

results["permeability_channel"] = results["R_pore"]**(-1)

globular_protein_density = 1.25#g/ml
experimental_data["d"] = experimental_data.apply(lambda _:estimate_protein_diameter(_.MW, globular_protein_density), axis = 1)
# boundary_z = 0
# results["permeability_bc"]=results.apply(lambda _:get_permeability_different_bc(boundary_z, _.permeability), axis = 1)
r_pore=r_dome = 26
wall_thickness = 26
dome_permeability = 4*np.pi*r_dome
thin_pore_permeabiity = empty_pore_permeability(1,r_pore,wall_thickness=26)
#%%
nups = ["Mac98A","Nup116"]
experimental_data_=experimental_data.loc[experimental_data.Oligomer == 1]
#experimental_data_ = experimental_data
################################################################################
################################################################################
fig, axs = plt.subplots(ncols = len(chi_PS), dpi = 600, sharey=True, sharex= True)
if len(chi_PS) == 1:
    axs_ = [axs]
else:
    axs_ = axs
results_ = results.loc[(results.mobility_model == model)]

for ax, (chi_PS_, result_) in zip(axs_, results_.groupby(by = "chi")):
    y = experimental_data_["Passage_Rate"]
    mpl_markers = ('o', 's', 'D')
    markers = itertools.cycle(mpl_markers)
    for nup in nups:
        x = experimental_data_[nup]
        s = experimental_data_["d"]
        ax.scatter(x,y, marker = next(markers), s=s*3, label = nup, color = "black", fc = "none")

    markers = itertools.cycle(mpl_markers)

    first_it =True
    for nup in nups:
        mark_protein = "mCherry"
        color = "red"
        y = experimental_data_.loc[experimental_data_.Protein == mark_protein,"Passage_Rate"]
        x = experimental_data_.loc[experimental_data_.Protein == mark_protein,nup]
        s = experimental_data_.loc[experimental_data_.Protein == mark_protein,"d"]
        if first_it: 
            label = mark_protein
            first_it = False
        else:
            label = None
        ax.scatter(x,y, color = color, marker = next(markers), s = s*3, label=label)
    
    markers = itertools.cycle(mpl_markers)
    first_it =True
    for nup in nups:
        mark_protein = "EGFP"
        color = "green"
        y = experimental_data_.loc[experimental_data_.Protein == mark_protein,"Passage_Rate"]
        x = experimental_data_.loc[experimental_data_.Protein == mark_protein,nup]
        s = experimental_data_.loc[experimental_data_.Protein == mark_protein,"d"]
        if first_it: 
            label = mark_protein
            first_it = False
        else:
            label = None
        ax.scatter(x,y, color = color, marker = next(markers), s = s*3, label=label)

    for d_, result__ in result_.groupby(by = "d"):
        permeability_key = "permeability_channel"
        #reference_permeability = result_.loc[(result_.chi_PC==reference_chi_PC) & (result__.d==reference_d),permeability_key].squeeze()
        reference_permeability = result__.loc[(result__.chi_PC==reference_chi_PC),permeability_key].squeeze()
        y = result__[permeability_key].squeeze()#*d_
        x = result__["PC_gel"].squeeze()
        y = y/reference_permeability
        ax.plot(
            x, y, 
            label = fr"$d = {d_}({d_*Kuhn_segment:.1f}"+r"\text{nm})$",
            #marker = next(markers),
            #markevery = 0.5,
            #markersize = 4,
        )
        normalized_dome_permeability = dome_permeability/reference_permeability
        ax.axhline(normalized_dome_permeability)
   
        # permeability_key = "permeability"
        # reference_permeability = result__.loc[result__.chi_PC==reference_chi_PC,permeability_key].squeeze()
        # y = result__[permeability_key].squeeze()#*d_
        # #y = result__["PC"].squeeze()
        # x = result__["PC_gel"].squeeze()
        # y = y/reference_permeability
        # ax.plot(x, y, linestyle = "--", color = ax.lines[-1].get_color(), linewidth= 0.5)

        # permeability_key = "permeability_longer"
        # reference_permeability = result__.loc[result__.chi_PC==reference_chi_PC,permeability_key].squeeze()
        # y = result__[permeability_key].squeeze()#*d_
        # #y = result__["PC"].squeeze()
        # x = result__["PC_gel"].squeeze()
        # y = y/reference_permeability
        # ax.plot(x, y, linestyle = "--", color = ax.lines[-1].get_color())

    # # Loop over each row in the DataFrame to add text
    # for i, row in experimental_data_.iterrows():
    #     y = row["Passage_Rate"]
    #     for nup in nups:
    #     #for nup in ["Nup116"]:
    #         text_key = "MW"
    #         x = row[nup]
    #         s = f"{row[text_key]:.1f}"
    #         ax.text(
    #             x*1.5,
    #             y*0.5,     # small offset in y
    #             s,
    #             ha='center',
    #             va='bottom',
    #             fontsize=5
    #         )

    ax.set_title(r"$\chi_{{PS}} = "+f"{chi_PS_}$")
    ax.set_ylim(1e-1, 1e5)
    ax.set_xlim(1e-2,1e4)
    ax.set_xlabel(r"$c_{\text{eq}}/c_0$")
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.grid()

    #ax.axvline(1, color = "red", linewidth = 0.5)


axs_[0].set_ylabel(r"$R_{\text{ref}} / R$, $J/J_{\text{ref}}$")

#axs_[-1].scatter([],[], marker = "*", color = "grey", label = "empty pore")
axs_[-1].legend( 
    bbox_to_anchor = [1.05, 0.05],
    loc = "lower left"
    )

#plt.tight_layout()
fig.set_size_inches(2.5*len(axs_)+0.5, 3)
#fig.savefig()
#%%