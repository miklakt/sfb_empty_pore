# %%
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('tableau-colorblind10')
from calculate_fields_in_pore import *

def empty_pore_permeability(D, r, L):
    return 2*D*r/(np.pi + 2*L/r)

%matplotlib qt
def plot_heatmap(fields, r_cut, z_cut, keys, **kwargs):
    from heatmap_explorer import plot_heatmap_and_profiles
    wall_thickness = fields["s"]
    l1 = fields["l1"]
    def cut_and_mirror(arr):
        cut = arr.T[0:r_cut, l1-z_cut:l1+wall_thickness+z_cut]
        return np.vstack((np.flip(cut), cut[:,::-1]))
    extent = [-z_cut-wall_thickness/2, z_cut+wall_thickness/2, -r_cut, r_cut]
    for key in keys:
        mask = cut_and_mirror(fields["walls"])
        fig = plot_heatmap_and_profiles(
            cut_and_mirror(fields[key]).T,
            y0=-r_cut,
            x0=-z_cut-wall_thickness/2,
            ylabel="$r$",
            xlabel = "$z$",
            zlabel=key,
            update_zlim=False,
            hline_y=int(z_cut+wall_thickness/2),
            vline_x=r_cut,
            mask = mask.T,
            **kwargs
            )
        fig.show()
        fig.savefig(f"kernel_figs/{key}.pdf")

def plot_permeability_z(ax, fields , L):
    perm = integrate_permeability_over_z(fields, L)
    perm = perm['permeability']/ perm['permeability_empty']
    s = fields["s"]
    l1 = fields["l1"]
    ylayers = fields["ylayers"]
    z = np.arange(-ylayers/2, ylayers/2, 1) 
    ax.plot(z, (fields["permeability_z"]/fields["permeability_z_empty"])**(-1))
    ax.set_yscale("log")
    ax.axvline(l1-ylayers/2, color = "grey")
    ax.axvline(l1+s+1-ylayers/2, color = "grey")
    ax.axvline(l1-L-ylayers/2, color = "red")
    ax.axvline(l1+s+L+1-ylayers/2, color = "red")
    ax.set_title(f"Permeability integrated over z-crossections, normalized by empty pore\n total: {perm:.3E}")
#%%
#%%
#Simulation results = 

#simulation_results = pd.DataFrame(
#    columns = ["d", "chi_PS", "chi_PC", "R"],
#    data = dict(
#            d=[8, 8, 16, 8, 8, 8, 8, 16, 8, 4, 6, 8, 12, ],
#            chi_PS = [0.4, 0.5, 0.5, 0.5, 0.3, 0.3, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5],
#            chi_PC = [-1.5, -1, -1, 0, -1.5, -1, -1.5, -1.5, -1.25, -1.25, -1.25, -1.25, -1.25],
#            R = [0.04, 0.341, 29, 127.604, 0.072, 1.286, 0.031, 0.591, 0.063, 0.130, 0.094, 0.062, 0.032]   
#        )
#)

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
   ]
)


simulation_empty_pore = pd.DataFrame(
    columns = ["d", "J_tot"],
    data = dict(
            d=[8, 16],
            #R = [0.203, 0.302]
            J_tot = [4.926, 3.311]
        )
)
# %%
a0, a1 = [0.70585835, -0.31406453]
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness









## Figure 1 
## Permeability on particle size at different solvent quality and particle affinity
d = np.arange(4, 34, 2)
chi_PS = [0.3, 0.4, 0.5, 0.6]
chi_PC = [-1.5, -1.25, -1, 0]
L=[0, 20]
#L=0

P_0 = empty_pore_permeability(1, pore_radius, wall_thickness)
P_0 = 1

empty_pore_perm = np.array([
    empty_pore_permeability(1, pore_radius-d_/2, wall_thickness) for d_ in d
    ])
empty_pore_thin_perm = np.array([
    empty_pore_permeability(1, pore_radius-d_/2, 0) for d_ in d
    ])

perm = []
for d_, chi_PS_, chi_PC_, L_ in itertools.product(d, chi_PS, chi_PC, L):
    print(d_, chi_PS_, chi_PC_)
    fields = calculate_fields(
        a0, a1, d=d_,
        chi_PC=chi_PC_, chi=chi_PS_,
        wall_thickness=wall_thickness,
        pore_radius=pore_radius,
        #**method
        )
    perm_ = integrate_permeability_over_z(fields, L_)
    perm.append(dict(
        d = d_,
        chi_PS = chi_PS_,
        chi_PC = chi_PC_,
        L = L_,
        perm = perm_['permeability']
        ))
perm = pd.DataFrame(perm)

fig, axs = plt.subplots(ncols = len(chi_PS), sharey=True)
for ax, (chi_PS_, perm_) in zip(axs, perm.groupby(by = "chi_PS")):
    for chi_PC_, perm__ in perm_.groupby(by = "chi_PC"):
        perm___ = perm__.loc[perm__.L==L[0]]
        x = perm___["d"].squeeze()
        y = perm___["perm"].squeeze()
        ax.plot(x, y/P_0, linewidth = 0.2)

        x = perm___["d"].squeeze()
        y = perm___["perm"].squeeze()
        ax.plot(x, 
                1/(1/y+1/empty_pore_thin_perm)/P_0, 
                #linewidth = 0.2,
                color = ax.lines[-1].get_color(),
                linestyle = "-",
                label = chi_PC_,
                )

        #perm___ = perm__.loc[perm__.L==L[1]]
        #x = perm___["d"].squeeze()
        #y = perm___["perm"].squeeze()
        #ax.plot(x, y/P_0, 
        #        #label = chi_PC_,
        #        linewidth = 0.2,
        #        linestyle = ":",
        #        color = ax.lines[-1].get_color())

        R = simulation_results.query(f"(chi_PS=={chi_PS_})&(chi_PC=={chi_PC_})")
        ax.scatter(
            R["d"], R["J_tot"]/P_0, 
            color = ax.lines[-1].get_color(),
            marker = "o",
            s = 50
            )


        ax.set_xlabel("d")
        
    ax.plot(
        d, 
        empty_pore_perm/P_0, 
        color = "black", 
        linestyle = "--",
        label = "$P_{theory}(d)$"
        )
    ax.plot(
        d, 
        empty_pore_thin_perm/P_0, 
        color = "black", 
        linestyle = ":",
        label = "$P_{theory, thin}(d)$"
        )
    

    ax.scatter(
        simulation_empty_pore["d"], 
        simulation_empty_pore["J_tot"]/P_0,
        color = "black",
        marker = "o",
        label = "numeric",
        s=50,
        )
    
    
    ax.set_yscale("log")
    #ax.set_xscale("log")
    #ax.axhline(1, color = "black", linestyle = "--")
    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    ax.set_ylim(1e-1, 1e2)
    #ax.grid()

#axs[0].set_ylabel("$P/P_{theory}(d=0)$")
axs[0].set_ylabel("$P$")
axs[-1].legend(title = "$\chi_{PC}$")#, bbox_to_anchor = [1.05, 0.75])
#%%
d = 8
chi_PC = -1.25
chi = 0.5
method = dict(
    exclude_volume = True, 
    truncate_pressure = False, 
    method = "convolve", 
    mobility_correction = "vol_average"
)

#%%
chi_pc_ = np.linspace(-2, 0)
phi_correction_factor = (a0 + a1*chi_pc_)
fig, ax = plt.subplots()

ax.plot(chi_pc_, phi_correction_factor)
ax.set_xlabel("$\chi_{PC}$")
ax.set_ylabel("$\phi_{corrected}/\phi$")
ax.axhline(1, color = 'black')
#%%
phi_ = np.linspace(0, 0.7)
chi_pc_ = np.arange(-2, 0.5, 0.5)
gamma_ = [gamma(chi, chi_pc__, phi_, a0, a1) for chi_pc__ in chi_pc_]
fig, ax = plt.subplots()

[ax.plot(phi_, gamma__, label = chi_pc__) for chi_pc__, gamma__ in zip(chi_pc_, gamma_)]
ax.set_xlabel("$\phi$")
ax.set_ylabel("$\gamma$")
ax.axhline(0, color = 'black')
ax.legend(title="$\chi_{PC}$")


#%%
fields = calculate_fields(
    a0, a1, d=d,
    chi_PC=chi_PC, chi=chi,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    #**method
    )
L=20
p=integrate_permeability_over_z(fields, L)
#%%
fig, ax = plt.subplots()
plot_permeability_z(ax, fields, L)
# %%
r_cut = 50
z_cut = 30
plot_heatmap(fields, r_cut, z_cut, keys = ["phi", "Pi", "gamma", "free_energy", "mobility", "conductivity"])
#%%
def find_critical_chi_PC(lb, rb, L, **kwargs):
    from scipy.optimize import brentq
    def fsolve(x : float):
        fields = calculate_fields(chi_PC = x, **kwargs)
        perm = integrate_permeability_over_z(fields, L)
        return perm['permeability']/perm['permeability_empty']-1
    try:
        x = brentq(fsolve, lb, rb, xtol = 0.01)
    except:
        x = np.nan
    return x
#%%
find_critical_chi_PC(
    lb = 0, rb =-3, L=20,
    a0=a0, a1=a1, d=d, chi=chi,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius
    )
#%%
#Figure 0
chi_PS = [0.2, 0.3, 0.4, 0.5, 0.6]
L = 20

d_crit = [find_critical_chi_PC(
    lb = 0, rb =-3, L=20,
    a0=a0, a1=a1, d=d, chi=chi_,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius
    ) for chi_ in chi_PS]
#%%
fig ,ax = plt.subplots()
ax.plot(chi_PS, d_crit)
ax.set_xlabel("$\chi_{PS}$")
ax.set_ylabel("$\chi_{PC}^{crit}$")
#%%
## Figure 2
## Permeability on particle affinity for different particle sizes
d = [2, 4, 8, 12]
chi_PS = [0.2, 0.5, 0.6, 0.8]
chi_PC = np.arange(-2, 0, 0.1)
L=20
perm = []
for d_, chi_PS_, chi_PC_ in itertools.product(d, chi_PS, chi_PC):
    print(d_, chi_PS_, chi_PC_)
    fields = calculate_fields(
        a0, a1, d=d_,
        chi_PC=chi_PC_, chi=chi_PS_,
        wall_thickness=wall_thickness,
        pore_radius=pore_radius,
        #**method
        )
    perm_ = integrate_permeability_over_z(fields, L)
    perm.append(dict(
        d = d_,
        chi_PS = chi_PS_,
        chi_PC = chi_PC_,
        perm = perm_['permeability']/perm_['permeability_empty']
        ))
perm = pd.DataFrame(perm)

fig, axs = plt.subplots(ncols = len(d), sharey=True)
for ax, (d_, perm_) in zip(axs, perm.groupby(by = "d")):
    for chi_PS_, perm__ in perm_.groupby(by = "chi_PS"):
        x = perm__["chi_PC"].squeeze()
        y = perm__["perm"].squeeze()
        ax.plot(x, y, label = chi_PS_)
    ax.set_xlabel("$\chi_{PC}$")
    ax.set_yscale("log")
    ax.axhline(1, color = "black")
    ax.set_title(d_)
    ax.set_ylim(1e-6, 1e6)
axs[0].set_ylabel("$P/P_{empty}$")
ax.legend(title = "$\chi_{PS}$")
#%%
d = [2, 4, 8, 16]
chi_PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
#chi_PS = np.arange(0, 0.7, 0.1)
chi_PC = np.arange(-2, 0.1, 0.1)
L=20
perm = []
for d_, chi_PS_, chi_PC_ in itertools.product(d, chi_PS, chi_PC):
    print(d_, chi_PS_, chi_PC_)
    fields = calculate_fields(
        a0, a1, d=d_,
        chi_PC=chi_PC_, chi=chi_PS_,
        wall_thickness=wall_thickness,
        pore_radius=pore_radius,
        #**method
        )
    perm_ = integrate_permeability_over_z(fields, L)
    perm.append(dict(
        d = d_,
        chi_PS = chi_PS_,
        chi_PC = chi_PC_,
        perm = perm_['permeability']/perm_['permeability_empty']
        ))
perm = pd.DataFrame(perm)

vmin = np.log10(min(perm["perm"]))
vmax = np.log10(max(perm["perm"]))
fig, axs = plt.subplots(ncols = len(d), sharey=True)
for ax, (d_, perm_) in zip(axs, perm.groupby(by = "d")):
    perm__ = pd.pivot(perm_, index = "chi_PC", columns = "chi_PS", values = "perm")
    extent = [min(perm__.columns)-0.05, max(perm__.columns)+0.05 , min(perm__.index)-0.05, max(perm__.index)+0.05]
    im = ax.imshow(np.log10(perm__.to_numpy()), origin = "lower", 
              extent = extent, 
              interpolation = "nearest",
              vmin = -3,
              vmax = 3
              )
    
    contour = ax.contour(
        perm__.to_numpy(), extent = extent, 
        origin = "lower", 
        colors = "0.15", 
        linewidth = 0.7, 
        levels = [1e-2, 1e-1, 1e2, 1e3]
        )
    
    contour = ax.contour(
        perm__.to_numpy(), extent = extent, 
        origin = "lower", 
        colors = "red", 
        linewidth = 0.7, 
        levels = [1]
        )
    
    ax.set_title(f"$d = {d_}$")
    ax.set_xlabel("$\chi_{PS}$")

    ax.set_xticks(perm__.columns[::-2])

axs[0].set_ylabel("$\chi_{PC}$")
axs[0].set_yticks(perm__.index[::2])
cbar = fig.colorbar(
        im, ax=axs, 
        extend='both', 
        shrink=0.6, 
        location = "bottom", 
        pad=0.2
        )
cbar.set_label("$P/P_{empty}$")
# %%

# %%
