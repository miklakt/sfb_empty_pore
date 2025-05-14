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

#from calculate_fields_in_pore import *

flux_vs_molar_weight = {}
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

def get_R_empty(        
        r_p,                #nm
        L,                  #nm
        d,                  #nm
        eta=0.00145,           #Pa*s
        T=293,              #K
        Haberman_correction = False
        ):
    k_B = 1.380649*1e-23   #J/K

    r_p_ = (r_p-d/2)*1e-9            #m^3
    L_ = (L+d)*1e-9                  #m^3
    d_ = d*1e-9                      #m^3
    eta_ = eta                       #Pa*s

    D_0_ = k_B * T / (3 * np.pi * eta_ * d_)  #m^2/s
    if Haberman_correction:
        from calculate_fields_in_pore import Haberman_correction_approximant
        K = float(Haberman_correction_approximant(d, r_p))
    else:
        K = 1.0
    R_0_ = K*L_ / (D_0_ * np.pi * r_p_**2) + 1 / (2 * D_0_ * r_p_) #s/m^3
    
    return R_0_


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

def get_translocation_empty_pore(        
        r_p,                #nm
        L,                  #nm
        d,                  #nm
        eta=0.00145,           #Pa*s
        T=293,              #K
        Haberman_correction = False,
        conc_gradient = 1.0, #µMol
        ):
    R_0 = get_R_empty(r_p, L, d, eta, T, Haberman_correction)
    NA = 6.022e23
    return conc_gradient/R_0*NA/1e3
    

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
    #Scaling relation
    #diameter_nm = 0.066*(MW_kDa*1000)**(0.37)*2
    return diameter_nm

def estimate_molecular_weight(diameter_nm, density=1.2):
    NA = 6.022e23  # Avogadro's number
    v_bar = 1 / density  # cm^3/g
    radius_nm = diameter_nm / 2.0
    volume_nm3 = (4.0 / 3.0) * np.pi * radius_nm**3
    volume_cm3 = volume_nm3 * 1.0e-21
    mass_one_molecule = volume_cm3 / v_bar
    mw_g_per_mol = mass_one_molecule * NA
    mw_kDa = mw_g_per_mol / 1000.0
    return mw_kDa

def tau_from_nc_ratio(conc_ratio, volume_ratio, time):
    """
    Calculates characteristic time of equilibration between two compartments
    Assuming at t=0 all c=1 in compartment1 and c=0 at compartment2
    The total amount of the solute is constant

    Parameters:
        conc_ratio (float): The ratio of concentration in compartments at a given time 
        volume_ratio : Volume ratio of the compartments
    """
    r = (conc_ratio*volume_ratio+1)/(1-conc_ratio)
    tau = time/np.log(r)
    return tau

def R_from_translocation_rate(
        translocation, 
        conc_gradient=1.0,     #µMol,
        eta=0.00145,           #Pa*s
        T=293,                 #K
        normalized = False,
        ):
    NA = 6.02214076*1e23
    R = NA/(translocation*conc_gradient/1e3)
    k_B = 1.380649*1e-23
    if normalized:
        R=R *k_B*T/eta
    return R

# def get_translocations_from_R(
#         R,
#         conc_gradient = 1.0#µMol
#         ):
    



#%%
data=pd.DataFrame(
    {
    "Probe": ["Fluorescein-Cys", "11 aa peptide", "Insulin", "Aprotinin", 
              "Profilin", "Ubiquitin", "z-domain", "Thioredoxin", 
              "Lactalbumin", "GFP", "PBP", "MBP", 
              "mCherry"],
    "MM":          [0.5, 1.4, 5.8, 6.5,
                    np.nan, 8.5, 8.2, 13.9,
                    14.2, 27, 37, 43, 
                    28],
    "stokes_r_nm": [0.67, 0.91, 1.19, 1.48,
                    1.65, 1.69, 1.71, 1.97,
                    2.07, 2.42, 2.75, 2.85, 
                    2.45],
    #Influx_rate
    "k":            [0.940, 0.53, 0.24, 0.086,
                    0.0548, 0.0356, 0.0401, 0.0203,
                    0.0144, 0.00205, 0.00026, 0.00022, 
                    5.7*1e-4],
    "qi":          [46, 26, 11.8, 4.25,
                    2.70, 1.75, 1.98, 1.00,
                    0.707, 0.1010, 0.0126, 0.0109,
                    0.028],
    "qi_std":      [9.2, 3.9, 1.31, 0.58,
                    0.40, 0.28, 0.28, np.nan,
                    0.012, 0.0140, 0.0071, 0.0028,
                    np.nan]
    })
data["tau"] = data["k"]**(-1)
data["Comment"] = ""
data.loc[data["Probe"] == 'Insulin', "Comment"] = "external data on MM"
data.loc[data["Probe"] == 'Lactalbumin', "Comment"] = "external data on MM"
data.loc[data["Probe"] == 'MBP', "Comment"] = "https://www.uniprot.org/uniprotkb/P0AG82/entry"

Mohr2009 = {
    "data" : data,
    "Culture" : "HeLa",
    "NuclearVolume" : 1130, #fL
    "NPCNumber" : 2770,
    "Reference" : "Mohr et al",
    "URL" : "https://doi.org/10.1038/emboj.2009.200",
}


flux_vs_molar_weight["Mohr2009"] = Mohr2009
#%%
data = pd.DataFrame(
    {
    "Probe":["GFP-HIS", "GFP-1PrA", "GFP-2PrA", "GFP-3PrA","GFP-4PrA", "GFP-6PrA", "GFP-1PrG", "GFP-2PrG"],
    "MM":[ 26.8, 34.2, 40.7,  46.8,  53.6,   66.8,  34.7,  42.3],
    "Rg":[ 2.5,  3.1,  3.3,   3.7,   3.9,    4.3,   3.2,   3.5],
    "tau":[15.0,   62.0,   114.0,   180.0,   252.0,    413.0,   20.0,    66.0], #Some values were extracted graphically from the Figure 3
    "tau_err":[1.4, np.nan, np.nan, np.nan, np.nan, 92, np.nan, np.nan],
    "D":[  9.33, 8.42, 7.1,   6.59,  6.17,   5.67, np.nan, np.nan]
    }
)
data["k"] = data["tau"]**(-1)
data["Comment"] = ""
Timney2016 = {
    "data" : data,
    "Culture" : "Saccharomyces cerevisiae",
    "NuclearVolume" : 4.8, #fL
    "CytoplasmVolume" : 60, #fL
    "NPCNumber" : 161,
    "Reference" : "Timney et al",
    "URL" : "https://doi.org/10.1083/jcb.201601004",
}
flux_vs_molar_weight["Timney2016"]= Timney2016

# %%
data = pd.DataFrame(
    {
    "Probe":["MG",   "MGM",   "MGM2",   "MGM4",   "MG2",   "MG3",   "MG4",   "MG5"],
    "MM": [  68.0,    109.0,   149.0,    230.0,    95.0,    122.0,   150.0,   177.0],
    "NC" :[np.nan,    0.85,    0.4,      0.21,     np.nan,  0.66,    0.53,    0.33]
    }
)
data["Comment"] = ""
Popken2015 = {
    "data" : data,
    "Culture" : "Saccharomyces cerevisiae",
    "NuclearVolume" : 4.8, #fL
    "CytoplasmVolume" : 60, #fL
    "VolRatio" : 4.8/60,
    "EquilibrationTime":1*60*60,
    "NPCNumber" : 161,
    "Reference" : "Popken et al",
    "URL" : "http://www.molbiolcell.org/cgi/doi/10.1091/mbc.E14-07-1175",
}
Popken2015["data"]["tau"] = tau_from_nc_ratio(
    data["NC"], 
    Popken2015["VolRatio"], 
    Popken2015["EquilibrationTime"]
    )
Popken2015["data"]["k"] = Popken2015["data"]["tau"]**(-1)
flux_vs_molar_weight["Popken2015"]= Popken2015
#%%
Ribbeck2001 = {}
#The nuclei we used have an ellipsoid shape with average radii of 8.0 ± 1.0, 6.4 ± 0.8 and 5.2 ± 0.7 μm in the three axes,
# a volume of ∼1130 μm3 and a surface area of ∼540 μm2
# They contain ∼2770 NPCs
data = pd.DataFrame({
    "Probe": [
        "Complex of core-M9 pentamers + five transportins",
        "Transportin",
        "BSA",
        "GFP",
        "NTF2",
        "NTF2 W7R"
    ],
    "MM": [630, 100, 68, 29, 29.5, 29.5],
    "Radius": [8.17, 4.13, 3.55, 2.36, 2.51, 2.51],
    "Translocations_": [28, 65, 0.1, 2, 250, 60],# through NPCs at Δc = 1 μM (NPC-1 s-1)
    "Translocations_empty": [120, 430, 540, 920, 850, 850] #Diffusion rate through a hypothetical ‘plugless’ NPC at Δc = 1 μM (pore−1 s−1)
})
data["Comment"] = ""
data.loc[data["Probe"] == 'BSA', "Comment"] = "<0.1"
data["R"] = 1/data["Translocations_"]*6.022e23/1e3

Ribbeck2001 = {
    "data" : data,
    "Culture" : "HeLa",
    "NuclearVolume" : 1130, #fL
    "NPCNumber" : 2770,
    "URL" : "https://doi.org/10.1093/emboj/20.6.1320",
    "Reference" : "Ribbeck et al",
}

flux_vs_molar_weight["Ribbeck2001"] = Ribbeck2001
# %%
def calculate_probe_diameter_from_molar_weight(density):
    for k, v in flux_vs_molar_weight.items():
        v["data"]["d"] = v["data"].apply(lambda _: estimate_protein_diameter(_["MM"], density=density), axis = 1)


for k, v in flux_vs_molar_weight.items():
    cytoplasm_volume = v.get("CytoplasmVolume")
    nuclear_volume = v.get("NuclearVolume")
    NPC_per_nucleus = v.get("NPCNumber")
    normalize = False
    if "k" in v["data"]:
        v["data"]["R"] = v["data"].apply(
            lambda _: R_from_k(
                _["k"], NPC_per_nucleus=NPC_per_nucleus,
                V_nucleus=nuclear_volume,
                V_cytoplasm=cytoplasm_volume,
                normalize=normalize
                ), axis = 1)

for k, v in flux_vs_molar_weight.items():
    v["data"]["Translocations"] = v["data"]["R"]**(-1)*6.022e23/1e3
# %%
if __name__=="__main__":
    Kuhn_segment = 0.76
    pore_radius = 26*Kuhn_segment
    L = pore_radius*2
    density = 1.2
    calculate_probe_diameter_from_molar_weight(density)

    empty_pore = {}
    MM = np.geomspace(1,700)
    d = estimate_protein_diameter(MM, density)
    translocations=get_translocation_empty_pore(pore_radius, L, d)
    #translocations=get_translocation_empty_pore(5, L, d)
    empty_pore["MM"] = MM
    empty_pore["d"] = d
    empty_pore["Translocations"] = translocations
    empty_pore["R"] = get_R_empty(pore_radius, L, d)
    #empty_pore["k"] = get_k_empty_pore(pore_radius, L, d, )

    show_text = False
    fig, ax = plt.subplots()
    markers = itertools.cycle(mpl_markers)
    Y_label = "Translocations"
    X_label = "MM"
    for k, v in flux_vs_molar_weight.items():
        x = v["data"][X_label]
        y = v["data"][Y_label]
        ax.scatter(x,y, label = v["Reference"], marker = next(markers))
        if show_text:
            for idx, row in v["data"].iterrows():
                x = row[X_label]
                y=  row[Y_label]
                s =row["Probe"]
                ax.text(x,y,s)
    ax.plot(empty_pore[X_label], empty_pore[Y_label], label = "Empty pore")
    ax.legend()
    #ax.plot(Ribbeck2001["data"]["MM"], Ribbeck2001["data"]["Translocations_empty"])

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1,650)

    fig.set_size_inches(3.5, 3.5)
# %%
