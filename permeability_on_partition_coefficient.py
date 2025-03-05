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
#%%
#d = np.arange(6, 24, 2)
d = [6]
#d = [6,4]
chi_PS = [0.6]
chi_PC = np.round(np.arange(-5, 0.2, 0.05),3)

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
results["permeability_channel"] = results["R_pore"]**(-1)
#%%
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

def chi_from_phi_binodal(phi):
    return (-np.log(1-phi) - phi)/(phi**2)

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
#%%
reference_chi_PC = 0.0
reference_d = 6
protein_density = 1.25#g/ml

#%%
#einstein_factor = 1/(3*np.pi*d[0])
#r_pore=26
#r_dome = r_pore*2
wall_thickness = 26
#dome_permeability = np.pi*r_dome*einstein_factor
#thin_pore_permeability = empty_pore_permeability(1,r_pore-d[0]/2,0)*einstein_factor
#thick_pore_permeability = empty_pore_permeability(1,r_pore,wall_thickness)*einstein_factor

def get_PC_in_gel(phi_gel, chi_PC, d):
    chi_gel=chi_from_phi_binodal(phi_gel)
    return PC_gel(phi_gel, chi_gel, chi_PC, d)

def get_thin_pore_permeability(d, r_pore = 26):
    return empty_pore_permeability(1.0,r_pore-d/2, 0)/(3*np.pi*d)

def get_dome_permeability(d, r_dome = 26*1.1):
    return np.pi*r_dome/(3*np.pi*d)
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
results_ = results.loc[(results.mobility_model == model)].sort_values(by = "chi_PC")

for ax, (chi_PS_, result_) in zip(axs_, results_.groupby(by = "chi")):
    y = experimental_data_["Passage_Rate"]
    mpl_markers = ('*', 'D')
    markers = itertools.cycle(mpl_markers)
    for nup in nups:
        x = experimental_data_[nup]
        #s = experimental_data_["d"]
        ax.scatter(
            x,y, 
        marker = next(markers), 
        #s=s*3,
        s=15,
        label = nup, 
        color = "black", 
        #fc = "none"
        )

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
        permeability_key = "permeability"
        reference_permeability = result_.loc[(result_.chi_PC==reference_chi_PC) & (result_.d==reference_d),permeability_key].squeeze()
        #reference_permeability = result__.loc[(result__.chi_PC==reference_chi_PC),permeability_key].squeeze()
        y = result__[permeability_key].squeeze()#*d_
        y = y/reference_permeability
        #x = result__["PC_gel"].squeeze()
        phi_gel_min = 0.2
        phi_gel_max = 0.4
        phi_gel_mean = (phi_gel_max+phi_gel_min)/2
        x = get_PC_in_gel(phi_gel_mean, result__["chi_PC"], d_)
        ax.plot(
            x, y, 
            label = fr"$d = {d_}({d_*Kuhn_segment:.1f}"+r"\text{nm})$",
            alpha = 1,
            color = "tab:blue"
            #marker = next(markers),
            #markevery = 0.5,
            #markersize = 4,
        )
        color_ = ax.lines[-1].get_color()

        if d_==reference_d:
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
            
            
            normalized_thin_pore_permeability = get_thin_pore_permeability(d_)/reference_permeability
            ax.axhline(normalized_thin_pore_permeability, color =  ax.lines[-1].get_color(), linestyle = "--", label = "thin pore")

            normalized_dome_permeability = get_dome_permeability(d_)/reference_permeability
            ax.axhline(normalized_dome_permeability, color =  ax.lines[-1].get_color(), linestyle = "--", linewidth = 0.3, label = "conductive sphere")


            ax.plot(x.iloc[::10], [10**3]*len(x.to_numpy()[::10]), marker = "|", linewidth = 0, color = "black")
            
            for xx, ss in zip(x.iloc[::10].squeeze(), result__["chi_PC"].iloc[::10].squeeze()):
                if ss<=-2.0: continue
                ax.text(xx*0.4, 10**3*1.1, s = f"{ss:.1f}")

        # x = result__["chi_PC"].squeeze()
        # ax2 = ax.twiny()
        # ax2.plot(x,y)
        # ax2.set_ylim(ax2.get_ylim()[::-1])
   
        #permeability_key = "permeability_channel"

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

    # Loop over each row in the DataFrame to add text

        # for i, row in experimental_data_.iterrows():
        #     y = row["Passage_Rate"]
        #     if y<7e-1:continue
        #     for nup in nups:
        #     #for nup in ["Nup116"]:
        #         text_key = "Protein"
        #         x = row[nup]
        #         #s = f"{row[text_key]:.1f}"
        #         s = row[text_key]
        #         ax.text(
        #             x*1.0,
        #             y*1.0,     # small offset in y
        #             s,
        #             ha='center',
        #             va='bottom',
        #             fontsize=5
        #         )

    ax.set_title(r"$\chi_{\text{PC}}$", pad = 18)
    ax.set_ylim(1e-1, 1e3)
    ax.set_xlim(1e-2,2e4)
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

fig.text(1,0.8, r"$\chi_{\text{PS}}^{\text{pore}} = "+f"{chi_PS_}$")
fig.text(1,0.7, r"$\chi_{\text{PS}}^{\text{gel}} = \{\Pi(\phi_{\text{gel}}) = 0\}$")

#plt.tight_layout()
#fig.set_size_inches(2.5*len(axs_)+0.5, 3)
#plt.tight_layout()
fig.set_size_inches(2.5, 2.5)
fig.savefig("fig/experimental_partitioning.svg")
#%%