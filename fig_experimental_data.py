#%%
from experimental_data import *
#%%
import itertools
import functools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rc


style.use('tableau-colorblind10')
get_palette_colors = lambda: itertools.chain([
 #'#FF800E',
 '#006BA4',
 '#ABABAB',
 '#595959',
 '#5F9ED1',
 #'#C85200',
 '#898989',
 '#A2C8EC',
 #'#FFBC79',
 '#CFCFCF'])
mpl_markers = ('+','o', '^', 's', 'D')

import calculate_fields_in_pore

Kuhn_segment = 0.76
a0 = 0.7
a1 = -0.3
L=52
r_pore=26
sigma = 0.02
alpha =  30**(1/2)
#alpha = 5
eta=0.00145            #Pa*s
T=293                  #K
NA = 6.02214076*1e23
k_B = 1.380649*1e-23
#d = np.arange(2.0, 32.0, 2)
#d = np.insert(d, 0, [0.5, 1])

#%%
from calculate_fields_in_pore import volume, surface, Pi, gamma
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

calculate_fields = functools.partial(
    calculate_fields_in_pore.calculate_fields,
        a0=a0, a1=a1, 
        wall_thickness = L, 
        pore_radius = r_pore,
        sigma = sigma,
        mobility_model_kwargs = {"prefactor":alpha},
    )

#R_0_no_vol_excl = np.array([calculate_fields_in_pore.empty_pore_permeability(1/(3*np.pi*d_), r_pore, L)**-1 for d_ in d])
#R_0 = np.array([calculate_fields_in_pore.empty_pore_permeability(1/(3*np.pi*d_), r_pore-d_/2, L+d_)**-1 for d_ in d])

#%%
def get_theory_for_given_density(density):
    calculate_probe_diameter_from_molar_weight(density)

    empty_pore = {}
    MM = np.geomspace(1,800, 500)
    d = estimate_protein_diameter(MM, density)
    translocations=get_translocation_empty_pore(pore_radius, L, d)
    #translocations=get_translocation_empty_pore(5, L, d)
    empty_pore["MM"] = MM
    empty_pore["d"] = d
    empty_pore["Translocations"] = translocations
    empty_pore["R"] = get_R_empty(pore_radius, L, d)
    #empty_pore["k"] = get_k_empty_pore(pore_radius, L, d, )

    chi_PS = 0.6
    chi_PC = 0
    inert_particles = {}
    inert_particles_d = np.arange(2, int(max(d))+1,2, dtype = float)
    inert_particles_d=np.insert(inert_particles_d, 0, [0.5, 1])
    inert_particles["d"] = inert_particles_d*Kuhn_segment
    inert_particles["MM"] = estimate_molecular_weight(inert_particles_d*Kuhn_segment, density)
    inert_particles_result = pd.DataFrame([calculate_fields(chi_PS=chi_PS, chi_PC=chi_PC, d=d_) for d_ in inert_particles_d])
    R = np.array(inert_particles_result["permeability"]**-1)
    inert_particles["R"] = R*eta/(k_B*T)
    inert_particles["Translocations"] = inert_particles["R"]**(-1)*NA/1e3

    return empty_pore, inert_particles
    
#%%
axis_label = {
    "MM":"MM, [kDa]",
    "Translocations":"Translocation through NPC\n"+r"at $\Delta c = 1 \mu \text{M}, [\text{s}^{-1}]$",
    "d":r"$d, [\text{nm}]$",
    "R":r"$R, [\text{m}^3/\text{s}]$",
}
Kuhn_segment = 0.76
pore_radius = 26*Kuhn_segment
L = pore_radius*2



show_text = False
fig, axs = plt.subplots(ncols = 2)
ax = axs[0]
ax.set_yscale("log")
ax.set_xscale("log")
mpl_markers = ('o', '+', '^', 's', 'D')
markers = itertools.cycle(mpl_markers)
Y_label = "Translocations"
X_label = "MM"
ax.set_xlabel(axis_label[X_label])
ax.set_ylabel(axis_label[Y_label])


density = 1.0
empty_pore, inert_particles = get_theory_for_given_density(density)
ax.plot(empty_pore[X_label], empty_pore[Y_label], label = "Empty pore", color = "k")
ax.plot(inert_particles[X_label], inert_particles[Y_label], label = "Inert particles", color = '#FF800E')
ax.set_xlim(min(empty_pore[X_label]),max(empty_pore[X_label]))

density = 1.4
empty_pore, inert_particles = get_theory_for_given_density(density)
ax.plot(empty_pore[X_label], empty_pore[Y_label], color = "k")
ax.plot(inert_particles[X_label], inert_particles[Y_label], color = '#FF800E')
ax.set_xlim(min(empty_pore[X_label]),max(empty_pore[X_label]))

color = get_palette_colors()

#flux_vs_molar_weight["Frey2018"]["data"] = flux_vs_molar_weight["Frey2018"]["data"].query("Probe in ['EGFP', 'mCherry']")

for k, v in flux_vs_molar_weight.items():
    if k == "Frey2018":
        data = flux_vs_molar_weight["Frey2018"]["data"].query("Probe in ['EGFP', 'mCherry']")
        ec_color_ = "red"
    elif k == "Ribbeck2001":
        data = flux_vs_molar_weight["Ribbeck2001"]["data"].query("Probe in ['GFP', 'BSA']")
    else:
        data = v["data"]
        color_ = next(color)
        ec_color_ = color_
    x = data[X_label]
    y = data[Y_label]
    ax.scatter(x,y, label = v["Reference"], marker = next(markers), ec = ec_color_, color = color_, zorder = 3)
    if show_text:
        for idx, row in data.iterrows():
            x = row[X_label]
            y=  row[Y_label]
            s =row["Probe"]
            ax.text(x,y,s, fontsize = 6)



ax.legend(
    #bbox_to_anchor = [0.0,1.0], 
    loc = "lower left"
    )
ax.grid()
ax.minorticks_off()
ax.set_ylim(5e-4, 3e3)

#fig.set_size_inches(2.5, 2.5)
################################################################################
ax = axs[1]
show_text = False
#fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xscale("log")
markers = itertools.cycle(mpl_markers)
Y_label = "Translocations"
#X_label = "MM"
ax.set_xlabel(r"$\left(\frac{c_{\text{in}}}{c_{\text{out}}}\right)_{\text{gel}}$")
#ax.set_ylabel(axis_label[Y_label])
#nups = ["Mac98A","Nup116", "Nsp1"]
nups = ["Mac98A","Nup116"]

data = Frey2018["data"].loc[Frey2018["data"]["d"]<5]

#ax.set_xlim(min(empty_pore[X_label]),max(empty_pore[X_label]))

y = data[Y_label]
mpl_markers = ('D')
markers = itertools.cycle(mpl_markers)
color = get_palette_colors()
for nup in nups:
    x = data[nup]
    ax.scatter(
        x,y,label = nup,
        marker = next(markers),
        color = next(color)
        )
    
    marked = data.query("Probe in ['EGFP', 'mCherry']")
    x_m = marked[nup]
    y_m = marked[Y_label]
    ax.scatter(
        x_m,y_m,
        marker = "D",
        ec = "red",
        fc = "none"
    )
    

#ax.plot(empty_pore[X_label], empty_pore[Y_label], label = "Empty pore", color = "k")
d = 6
phi_gel = 0.3
empty_pore_line = get_translocation_empty_pore(
        r_p = pore_radius,
        L=L,
        d=d*Kuhn_segment,
        #Frey2018["NPCNumber"],
        #Frey2018["NuclearVolume"],
        #Haberman_correction=True
        )

chi_PS = 0.6
chi_PCs = np.arange(0.2, -2.4, -0.1)
theoretical_particles = {}
theoretical_particles["d"] = d*Kuhn_segment
theoretical_particles["MM"] = estimate_molecular_weight(d*Kuhn_segment, density)
theoretical_particles_result = pd.DataFrame([calculate_fields(chi_PS=chi_PS, chi_PC=chi_PC, d=d) for chi_PC in chi_PCs])
R = np.array(theoretical_particles_result["permeability"]**-1)
theoretical_particles["R"] = R*eta/(k_B*T)
theoretical_particles["Translocations"] = theoretical_particles["R"]**(-1)*NA/1e3
theoretical_particles["PC"] = [PC_gel(phi_gel, chi_PS, chi_PC, d) for chi_PC in chi_PCs]

ax.plot(theoretical_particles["PC"], theoretical_particles[Y_label], color = '#FF800E', linewidth = 1.5, linestyle = "--")

theoretical_particles_result = pd.DataFrame([calculate_fields(chi_PS=chi_PS, chi_PC=chi_PC, d=d, stickiness=True) for chi_PC in chi_PCs])
R = np.array(theoretical_particles_result["permeability"]**-1)
theoretical_particles["R"] = R*eta/(k_B*T)
theoretical_particles["Translocations"] = theoretical_particles["R"]**(-1)*NA/1e3

ax.plot(theoretical_particles["PC"], theoretical_particles[Y_label], color = '#FF800E', linewidth = 2)

ax.axhline(
    empty_pore_line, 
    color =  "black", 
    linestyle = "-", 
    label = "empty pore",
    linewidth = 2
    )

ymax=5e3
for xx, ss in zip(theoretical_particles["PC"][2::5], chi_PCs[2::5]):
    if ss<=-2.2: continue
    if ss>0.0: continue
    ax.text(xx*0.4, ymax*1.1, s = f"{ss:.1f}")
    ax.scatter([xx],[ymax],marker = "|", color = "k")



ax.legend(
    #bbox_to_anchor = [0.0,1.0], 
    loc = "lower left"
    )
ax.grid()
ax.set_ylim(1e-1, ymax)
ax.set_xlim(2e-2, 2e4)
ax.minorticks_off()

fig.set_size_inches(6, 2.5)

fig.savefig("fig/experimental_data.svg", dpi = 600)
#%%