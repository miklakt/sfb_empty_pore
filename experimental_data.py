#%%
from io import StringIO
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rc

#rc('text',usetex=True)
#rc('text.latex', preamble=r'\usepackage{color}')
style.use('tableau-colorblind10')

#from calculate_fields_in_pore import *

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
flux_vs_molar_weight = {}
data=pd.DataFrame(
    {
    "Probe": ["Fluorescein-Cys", "11 aa peptide", "Insulin", "Aprotinin", 
              "Profilin", "Ubiquitin", "z-domain", "Thioredoxin", 
              "Lactalbumin", "GFP", "PBP", "MBP",],
    "MM":          [0.5, 1.4, 5.8, 6.5,
                    np.nan, 8.5, 8.2, 13.9,
                    14.2, 27, 37, 43,],
    "stokes_r_nm": [0.67, 0.91, 1.19, 1.48,
                    1.65, 1.69, 1.71, 1.97,
                    2.07, 2.42, 2.75, 2.85,],
    #Influx_rate
    "k":            [0.940, 0.53, 0.24, 0.086,
                    0.0548, 0.0356, 0.0401, 0.0203,
                    0.0144, 0.00205, 0.00026, 0.00022,],
    "qi":          [46, 26, 11.8, 4.25,
                    2.70, 1.75, 1.98, 1.00,
                    0.707, 0.1010, 0.0126, 0.0109,],
    "qi_std":      [9.2, 3.9, 1.31, 0.58,
                    0.40, 0.28, 0.28, np.nan,
                    0.012, 0.0140, 0.0071, 0.0028,]
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
    "Translocations_empty": [120, 430, 540, 920, 850, 850] # Diffusion rate through a hypothetical 'plugless' NPC at Δc = 1 μM (pore−1 s−1)
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
#%%
csv_data="""Probe,Arg/Lys,Charge_pH7.4,Passage_Rate,Mac98A,Nup116,Nsp1,MM,MM_err,Oligomer,Plasmid
yNTF2 (dimer),10R/6K,-15.0,600.0,290.0,1400.0,800.0,28.0,,1.0,pSF360
rNTF2 (dimer),4R/8K,-14.6,680.0,3200.0,13000.0,890.0,28.0,,1.0,pDG2121
mCherry,8R/24K,-6.4,1.0,0.05,0.09,0.4,28.0,,1.0,pSF779
EGFP,6R/21K,-8.1,3.5,0.11,0.33,0.98,28.0,0.6%,1.0,pSF1526
shGFP2,11R/24K,-6.2,0.47,0.05,0.05,0.24,29.0,1.4%,1.0,pSF1438
sinGFP1,1R/34K,-6.4,0.23,0.05,0.05,0.18,28.0,1.5%,1.0,pSF2893
sinGFP4a,1R/34K,-11.4,0.1,0.05,0.05,,28.0,0.5%,1.0,pDG2754
efGFP_0W,6R/20K,-10.1,6.0,0.09,0.42,3.0,28.0,1.6%,1.0,pSF2646
efGFP_3W,6R/20K,-10.1,82.0,1.5,14.0,104.0,33.0,0.5%,1.2,pSF2647
efGFP_5W,6R/19K,-11.1,87.0,2.2,15.0,170.0,56.0,0.3%,2.0,pSF2648
efGFP_8F,6R/18K,-11.0,43.0,8.3,51.0,102.0,170.0,1.9%,6.0,pSF2653
efGFP_8L,6R/18K,-11.0,33.0,1.8,10.0,90.0,59.0,0.2%,2.2,pSF2651
efGFP_8I,6R/18K,-11.0,68.0,2.9,23.0,102.0,56.0,0.2%,2.1,pSF2650
efGFP_8M,6R/18K,-11.0,110.0,3.0,17.0,92.0,48.0,0.5%,1.7,pSF2652
efGFP_8R,14R/18K,-3.0,19.0,0.5,1.9,5.8,30.0,0.7%,1.0,pSF2892
sffrGFP4,26R/1K,-7.7,160.0,14.0,50.0,9.8,29.0,1.8%,1.0,pDG2805
sffrGFP4_18xR/K,8R/19K,-8.1,3.8,0.12,0.31,0.76,28.0,2.2%,1.0,pSF2884
sffrGFP4_25xR/K,1R/26K,-9.2,1.6,0.06,0.1,0.54,27.0,1.8%,1.0,pSF2885
sffrGFP4_duplicate,26R/1K,-7.7,160.0,14.0,50.0,,29.0,0.3%,1.0,pDG2805
sffrGFP5,26R/1K,-15.7,64.0,0.67,5.5,,28.0,1.0%,1.0,pDG2806
sffrGFP6,26R/1K,-1.7,280.0,100.0,160.0,,29.0,1.6%,1.0,pDG2713
sffrGFP7,26R/1K,-0.3,310.0,200.0,200.0,,29.0,1.4%,1.0,pDG2715
GFP_MaxR_5W,23R/4K,-7.8,830.0,2100.0,4000.0,690.0,140.0,0.4%,5.0,pSF2903
GFP_MaxR_8i,25R/1K,-8.7,1300.0,2000.0,4100.0,640.0,47.0,1.6%,1.7,pSF2905
GFPNTR_2B7,25R/1K,-10.7,1600.0,1200.0,1600.0,,65.0,1.7%,2.4,pDG2718
GFPNTR_7B3,25R/1K,-9.7,1700.0,1700.0,1400.0,,42.0,2.0%,1.6,pDG2798"""

csv_buffer = StringIO(csv_data)

data = pd.read_csv(csv_buffer)
_mCherry_influx = 5.7*1e-4
_nuclear_volume = 1130
_NPC_number = 2770

data["k"] = data["Passage_Rate"]*_mCherry_influx
data["tau"] = data["k"]**(-1)

data["R"] = data.apply(
    lambda _: R_from_k(
        _["k"], NPC_per_nucleus=_NPC_number,
        V_nucleus=_nuclear_volume,
        V_cytoplasm=None,
        normalize=False
        ), axis = 1)
data["Translocations"] = data["R"]**(-1)*6.022e23/1e3

data = data.loc[data["MM"]<35]

Frey2018 = {
    "data" : data,
    "Culture" : "HeLa",
    "NuclearVolume" : _nuclear_volume, #fL
    "NPCNumber" : _NPC_number,
    "URL" : "https://doi.org/10.1016/j.cell.2018.05.045",
    "Reference" : "Frey et al",
}

flux_vs_molar_weight["Frey2018"] = Frey2018
# %%
def calculate_probe_diameter_from_molar_weight(density):
    for k, v in flux_vs_molar_weight.items():
        v["data"]["d"] = v["data"].apply(lambda _: estimate_protein_diameter(_["MM"], density=density), axis = 1)
    #attractive particles separately
    Frey2018["data"]["d"]= Frey2018["data"].apply(lambda _: estimate_protein_diameter(_["MM"], density=density), axis = 1)


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
    axis_label = {
        "MM":"MM, [kDa]",
        "Translocations":"Translocation through NPC\n"+r"at $\Delta c = 1 \mu \text{M}, [\text{s}^{-1}]$",
        "d":r"$d, [\text{nm}]$",
        "R":r"$R, [\text{m}^3/\text{s}]$",
    }
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
    ax.set_yscale("log")
    ax.set_xscale("log")
    mpl_markers = ('o', '+', 'x', 's', '*')
    markers = itertools.cycle(mpl_markers)
    Y_label = "Translocations"
    X_label = "MM"
    ax.set_xlabel(axis_label[X_label])
    ax.set_ylabel(axis_label[Y_label])

    ax.set_xlim(min(empty_pore[X_label]),max(empty_pore[X_label]))

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
    ax.plot(empty_pore[X_label], empty_pore[Y_label], label = "Empty pore", color = "k")
    ax.legend(bbox_to_anchor = [1.0,0.5])
    ax.grid()
    
    fig.set_size_inches(3.5, 3.5)
    #%%
    show_text = False
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xscale("log")
    markers = itertools.cycle(mpl_markers)
    Y_label = "Translocations"
    #X_label = "MM"
    ax.set_xlabel(r"$\left(\frac{c_{\text{in}}}{c_{\text{out}}}\right)_{\text{gel}}$")
    ax.set_ylabel(axis_label[Y_label])
    nups = ["Mac98A","Nup116", "Nsp1"]

    data = Frey2018["data"].loc[Frey2018["data"]["d"]<5]

    #ax.set_xlim(min(empty_pore[X_label]),max(empty_pore[X_label]))

    y = data[Y_label]
    mpl_markers = ('*')
    markers = itertools.cycle(mpl_markers) 
    for nup in nups:
        x = data[nup]
        ax.scatter(
            x,y,label = nup,
            marker = next(markers),
            )
    #ax.plot(empty_pore[X_label], empty_pore[Y_label], label = "Empty pore", color = "k")
    d = 6*Kuhn_segment
    empty_pore_line = get_translocation_empty_pore(
            r_p = pore_radius,
            L=L,
            d=d,
            #Frey2018["NPCNumber"],
            #Frey2018["NuclearVolume"],
            #Haberman_correction=True
            )
    ax.axhline(
        empty_pore_line, 
        color =  "black", 
        linestyle = "-", 
        label = "empty pore"
        )
    ax.legend(bbox_to_anchor = [1.0,0.5])
    ax.grid()
    fig.set_size_inches(3.5, 3.5)
# %%
