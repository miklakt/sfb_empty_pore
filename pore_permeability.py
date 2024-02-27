# %%
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('tableau-colorblind10')
mpl_markers = ('o', '+', 'x', 's', 'D')

from calculate_fields_in_pore import *
from pickle_cache import pickle_lru_cache

def empty_pore_permeability(D, r, L):
    return 2*D*r/(np.pi + 2*L/r)

def D_effective(P, r, L):
    return P*(np.pi + 2*L/r)/(2*r)

def plot_permeability_z(ax, fields , L):
    perm = integrate_permeability_over_z(fields, L)
    perm = perm['permeability']
    s = fields["s"]
    l1 = fields["l1"]
    ylayers = fields["ylayers"]
    z = np.arange(-ylayers/2, ylayers/2, 1) 
    ax.plot(z, (fields["permeability_z"])**(-1))
    ax.set_yscale("log")
    ax.axvline(l1-ylayers/2, color = "grey")
    ax.axvline(l1+s+1-ylayers/2, color = "grey")
    ax.axvline(l1-L-ylayers/2, color = "red")
    ax.axvline(l1+s+L+1-ylayers/2, color = "red")
    ax.set_title(f"Permeability integrated over z-crossections, \n total: {perm:.3E}")

@pickle_lru_cache(purge_cache=False)
def find_critical_chi_PC(L=0, limit = False, **kwargs):
    lb = -3
    rb = 0
    from scipy.optimize import brentq
    def fsolve(x : float):
        fields = calculate_fields(chi_PC = x, **kwargs)
        perm = integrate_permeability_over_z(fields, L)
        r_apparent =  fields["r"]-kwargs["d"]/2
        if limit:
            perm["permeability"] = (perm["permeability"]**(-1)  + empty_pore_permeability(D=1, r =r_apparent, L = 0)**(-1))**(-1)
        return perm['permeability']/empty_pore_permeability(D=1, r =r_apparent, L = fields["s"])-1
    x = brentq(fsolve, lb, rb, xtol = 0.01)
    return x

#%%
simulation_results = pd.DataFrame(
   columns =    [   "d",   "chi_PS",    "chi_PC",   "J_tot",    "J_tot_err"],
   data = [     [   4,      0.5,        -1.0,       4.936,      0.014],
                [   4,      0.5,        -1.25,      7.716,      0.018],
                [   6,      0.5,        -1.25,     10.596,      0.028],
                [   8,      0.5,        -1.25,     16.131,      0.042],
                [  12,      0.5,        -1.25,     31.022,      0.080],
                [  16,      0.5,        -1.25,     38.650,      0.376],
                [  24,      0.5,        -1.25,      27.00,      10.00],
                [   4,      0.5,        -1.0,       4.933,      0.011],
                [   6,      0.5,        -1.0,       3.943,      0.010],
                [   8,      0.5,        -1.0,       2.954,      0.007],
                [  12,      0.5,        -1.0,       0.844,      0.002],
                [   4,      0.4,        -1.25,      6.756,      0.016],
                [   6,      0.4,        -1.25,      7.065,      0.017],
                [   8,      0.4,        -1.25,      7.145,      0.017],
                [  10,      0.4,        -1.25,      6.118,      0.015],
                [  12,      0.4,        -1.25,      3.731,      0.010],
                [   8,      0.4,         -1.0,      1.392,      0.004],
                [   4,      0.4,         -1.5,     10.037,      0.023],
                [   8,      0.4,         -1.5,     25.761,      0.060],
                [   4,      0.3,         -1.5,      8.562,      0.021],
                [   8,      0.3,         -1.5,     14.102,      0.034],
                [  12,      0.3,         -1.5,     14.362,      0.036],
                [  14,      0.3,         -1.5,      7.362,      0.036],
                [  16,      0.3,         -1.5,      1.898,      0.007],
                [  10,      0.6,         -1.0,     11.168,      0.029]
   ]
)

simulation_empty_pore = pd.DataFrame(
    columns = ["d", "J_tot"],
    data = dict(
            d=[4, 8, 16, 24],
            #R = [0.203, 0.302]
            J_tot = [5.927, 4.926, 3.311, 2.063]
        )
)
# %%
a0, a1 = [0.70585835, -0.31406453]
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness
def calculate_permeability(
        d, chi_PS, chi_PC, L,
        exclude_volume,#=True 
        truncate_pressure,#=False 
        method,#= "convolve", 
        mobility_correction,# = "vol_average",
        mobility_model,#, = "Rubinstein",
        mobility_model_kwargs:dict):
    
    fields = calculate_fields(
        a0=a0, a1=a1, d=d,
        chi_PC=chi_PC, chi=chi_PS,
        wall_thickness=wall_thickness,
        pore_radius=pore_radius,
        exclude_volume=exclude_volume,
        mobility_correction=mobility_correction,
        mobility_model = mobility_model,
        truncate_pressure=truncate_pressure,
        method = method,
        **mobility_model_kwargs
        )
    result = dict(
        a0=a0, a1=a1, d=d, L=L,
        chi_PC=chi_PC, chi=chi_PS,
        wall_thickness=wall_thickness,
        pore_radius=pore_radius,
        exclude_volume=exclude_volume,
        mobility_correction=mobility_correction,
        mobility_model = mobility_model,
        truncate_pressure=truncate_pressure,
        method = method,
        mobility_model_kwargs = mobility_model_kwargs
    )
    result["thin_empty_pore"] = empty_pore_permeability(1, pore_radius, 0)
    result["thick_empty_pore"] = empty_pore_permeability(1, pore_radius-d/2, wall_thickness)
    result["permeability"] = integrate_permeability_over_z(fields, L)
    return result
#%%
## Figure 1 
## Permeability on particle size at different solvent quality and particle affinity
d = np.arange(4, 34, 2)
#d =[8 ,10, 12 ,]
chi_PS = [0.4, 0.5, 0.6]
chi_PC = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5]
chi_PC_color = [-1.5, -1.25, -1]
L=[0, 20]

results = []
for d_, chi_PS_, chi_PC_, L_ in itertools.product(d, chi_PS, chi_PC, L):
    print(d_, chi_PS_, chi_PC_, L_)
    result = calculate_permeability(
        d_, chi_PS_, chi_PC_, L_,
        exclude_volume=True,
        truncate_pressure=False,
        method= "convolve", 
        # mobility_correction= "vol_average",
        # mobility_model = "Rubinstein",
        # mobility_model_kwargs = {"prefactor":1}
        mobility_correction = "no_mobility",
        mobility_model = "Phillies",
        mobility_model_kwargs = {"nu":0.7, "beta":8}
        )
    results.append(result)
results = pd.DataFrame(results)

#%%
markers = itertools.cycle(mpl_markers)
fig, axs = plt.subplots(ncols = len(chi_PS), sharey=True)
#results_ = results.loc[results.mobility_model == "Rubinstein"]
results_ = results.loc[results.mobility_model == "Phillies"]
for ax, (chi_PS_, result_) in zip(axs, results_.groupby(by = "chi")):
    for chi_PC_, result__ in result_.groupby(by = "chi_PC"):
        
        result___ = result__.loc[result__.L==L[1]]
        x = result___["d"].squeeze()
        #y = result___["permeability"].squeeze()
        y = (result___["permeability"]/result___["thick_empty_pore"]).squeeze()
        if chi_PC_ in chi_PC_color:
            plot_kwargs = dict(
                label = fr"$P_{{channel}}\left(\chi_{{PC}} = {chi_PC_}\right)$",
                        #marker = next(markers),
                        #markevery = 0.2
                        )
        else:
            plot_kwargs = dict(linewidth = 0.1,
                               color ="black"
                               )
        ax.plot(
            x, y, 
            **plot_kwargs
            )
        


        # x = perm___["d"].squeeze()
        # y = perm___["perm"].squeeze()
        # ax.plot(x, 
        #          1/(1/y+1/empty_pore_thin_perm), 
        #          #linewidth = 0.2,
        #          #color = ax.lines[-1].get_color(),
        #          linestyle = "-",
        #          label = chi_PC_,
        #          )



        # R = simulation_results.query(f"(chi_PS=={chi_PS_})&(chi_PC=={chi_PC_})")
        # ax.scatter(
        #     R["d"], R["J_tot"], 
        #     color = ax.lines[-1].get_color(),
        #     marker = "o",
        #     s = 50
        #     )


        ax.set_xlabel("d")
    
    # ax.plot(
    #     d, 
    #     empty_pore_perm, 
    #     color = "black", 
    #     linestyle = "--",
    #     label = "$P_{rayleigh}(d)$"
    #     )
    ax.plot(
        d, 
        empty_pore_permeability(1, pore_radius-d/2, L=0)/empty_pore_permeability(1, pore_radius-d/2, L=wall_thickness),
        color = "black", 
        linestyle = ":",
        label = "$P_{thin}(d)$"
        )
    
    ax.plot(
        d, 
        #empty_pore_permeability(1, pore_radius-d/2, L=wall_thickness),
        np.ones_like(d), 
        color = "black", 
        linestyle = "--",
        label = "$P_{thick}(d)$"
        )
    
    # ax.scatter(
    #     simulation_empty_pore["d"], 
    #     simulation_empty_pore["J_tot"],
    #     color = "black",
    #     marker = "o",
    #     label = "simulation",
    #     s=50,
    #     )
    
    ax.set_yscale("log")
    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    ax.set_ylim(1e-2, 1e3)

#axs[0].set_ylabel("$P/P_{theory}(d=0)$")
axs[0].set_ylabel("$P/P_{0}$")
axs[-1].legend(
    #title = "$\chi_{PC}$", 
    bbox_to_anchor = [1.0, 0.85]
    )

fig.text(s = r"$P_{0} = \frac{2Dr_{pore}}{\pi}$", x = 0.7, y = 0.2, ha = "left", fontsize = 14)

plt.tight_layout()
fig.set_size_inches(7, 3.5)
#fig.savefig("/home/ml/Desktop/permeability.pdf")
#%%
phi = np.linspace(0.01,1)
D = mobility_Phillies(phi, 8, 0.7)
plt.plot(phi, D, label ="Phillies")
plt.yscale("log")

phi = np.linspace(0.01,1)
d=8
D = mobility_Rubinstein(phi, 1, d)
plt.plot(phi, D, label ="Rubinstein")
plt.yscale("log")

plt.xlabel("$\phi$")

plt.ylabel("$D/D_{0}$")

plt.legend()
# %%
a0, a1 = [0.70585835, -0.31406453]
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness
markers = itertools.cycle(mpl_markers)
## Figure 1 
## Permeability on particle size at different solvent quality and particle affinity
d = np.arange(4, 34, 2)
chi_PS = [0.4, 0.5, 0.6]
chi_PC = [-1.5, -1.25, -1]
L=0


#empty_pore_perm = np.array([
#    empty_pore_permeability(1, pore_radius-d_/2, wall_thickness) for d_ in d
#    ])#/P0

#empty_pore_thin_perm = np.array([
#    empty_pore_permeability(1, pore_radius-d_/2, 0) for d_ in d
#    ])#/P0

perm = []
for d_, chi_PS_, chi_PC_ in itertools.product(d, chi_PS, chi_PC):
    print(d_, chi_PS_, chi_PC_)
    fields = calculate_fields(
        a0, a1, d=d_,
        chi_PC=chi_PC_, chi=chi_PS_,
        wall_thickness=wall_thickness,
        pore_radius=pore_radius,
        #prefactor = 10
        #**method
        )
    perm_ = integrate_permeability_over_z(fields, L=0)
    perm.append(dict(
        d = d_,
        chi_PS = chi_PS_,
        chi_PC = chi_PC_,
        perm = perm_['permeability']
        ))
perm = pd.DataFrame(perm)

perm["D_effective"] = D_effective(perm["perm"], pore_radius-perm["d"]/2, wall_thickness)
#%%
fig, axs = plt.subplots(ncols = len(chi_PS), sharey=True)
for ax, (chi_PS_, perm_) in zip(axs, perm.groupby(by = "chi_PS")):
    markers = itertools.cycle(mpl_markers)
    for chi_PC_, perm__ in perm_.groupby(by = "chi_PC"):
        #perm___ = perm__.loc[perm__.L==L[0]]
        x = perm__["d"].squeeze()
        y = perm__["D_effective"].squeeze()#/P0
        ax.plot(
            x, y, 
            #linewidth = 0.2,
            label = fr"$D_{{eff}}\left(d, \chi_{{PC}} = {chi_PC_}\right)$",
            marker = next(markers),
            markevery = 0.2
            )
        


        #x = perm___["d"].squeeze()
        #y = perm___["perm"].squeeze()
        # ax.plot(x, 
        #         1/(1/y+1/empty_pore_thin_perm), 
        #         #linewidth = 0.2,
        #         #color = ax.lines[-1].get_color(),
        #         linestyle = "-",
        #         label = chi_PC_,
        #         )

        # R = simulation_results.query(f"(chi_PS=={chi_PS_})&(chi_PC=={chi_PC_})")

        # ax.scatter(
        #     R["d"], R["J_tot"], 
        #     color = ax.lines[-1].get_color(),
        #     marker = "o",
        #     s = 50
        #     )


        ax.set_xlabel("d")
    
    # ax.plot(
    #     d, 
    #     empty_pore_perm, 
    #     color = "black", 
    #     linestyle = "--",
    #     label = "$P_{rayleigh}(d)$"
    #     )
    # ax.plot(
    #     d, 
    #     empty_pore_thin_perm, 
    #     color = "black", 
    #     linestyle = ":",
    #     label = "$P_{convergent}(d)$"
    #     )
    
    # ax.scatter(
    #     simulation_empty_pore["d"], 
    #     simulation_empty_pore["J_tot"],
    #     color = "black",
    #     marker = "o",
    #     label = "simulation",
    #     s=50,
    #     )
    
    ax.set_yscale("log")
    #ax.set_xticks([4,8,16,32])
    #ax.set_xticklabels(["4","8","16","32"], rotation = "vertical")
    #ax.set_xscale("log")
    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    ax.set_ylim(1e-2, 1e3)

    ax.axhline(1, color = "black", linewidth = 0.7)

    ax.set_xlim(0)

#axs[0].set_ylabel("$P/P_{theory}(d=0)$")
axs[0].set_ylabel("$D_{eff}/D_{0}$")
axs[-1].legend(
    #title = "$\chi_{PC}$", 
    bbox_to_anchor = [1.0, 0.85]
    )

#fig.text(s = r"$P_{0} = \frac{2Dr_{pore}}{\pi}$", x = 0.7, y = 0.2, ha = "left", fontsize = 14)

#fig.text(s = r"$D_{eff}/D_{0} = P_{channel} \cdot \frac{\pi+2s/(r-d/2)}{2(r-d/2)}$", x = 0.7, y = 0.1, ha = "left", fontsize = 14)
#fig.text(s =r"$P_{channel} = \left[\int_{-s/2}^{s/2} \left( \int_{0}^{r_{pore}} D e^{-\Delta F / kT} r dr \right)^{-1} dz \right]^{-1}$", x = 0.7, y = 0.2, ha = "left", fontsize = 14)


plt.tight_layout()
fig.set_size_inches(7, 3.5)
#fig.savefig("/home/ml/Desktop/permeability.pdf")
#%%
#%%
#Figure critical chi_PC value
d = [4, 6, 8, 12, 16, 24, 32]
chi_PS = [0.2, 0.3, 0.4, 0.5, 0.6]

chi_crit = np.array([[find_critical_chi_PC(
    #L=20,
    #limit = True,                      
    a0=a0, a1=a1, d=d_, chi=chi_,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius
    ) for chi_ in chi_PS] for d_ in d]
)

fig ,ax = plt.subplots()
for d_, chi_crit_ in zip(d, chi_crit):
    ax.plot(chi_PS, chi_crit_, label = d_)
    ax.text(chi_PS[0], chi_crit_[0], d_, color = ax.lines[-1].get_color(), ha = 'right')
    ax.set_xlabel("$\chi_{PS}$")
    ax.set_ylabel("$\chi_{PC}^{crit}$")
ax.legend(title = "d")
ax.grid()

#fig.savefig("tex/fig/fig2.pdf")
# %%
fig, ax = plt.subplots()
for chi_PS_, chi_crit_ in zip(chi_PS, chi_crit.T):
    ax.plot(d, chi_crit_, label = chi_PS_)
    ax.text(d[-1], chi_crit_[-1], chi_PS_, color = ax.lines[-1].get_color(), ha = 'left', va = "center")
    ax.set_xlabel("$d$")
    ax.set_ylabel("$\chi_{PC}^{crit}$")
ax.legend(title = "$\chi_{PS}$")
ax.grid()
fig.savefig("tex/fig/fig3.pdf")
# %%
