#%%
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import itertools

import numpy as np
import utils
import pandas as pd
from fit_gamma_model import gamma, free_energy_cylinder
#%%
s = 52
r = 26
ph =[8, 12, 16]
pw =ph
sigma = 0.02
chi_PS = [0.3, 0.4, 0.5, 0.6, 0.7]
chi_PC = [-1.5, -1.0, 0.0]

master = pd.read_pickle("pkl/reference_table.pkl")
#master = master.loc[master["comment"] == "grown_from_small"]
master_empty = pd.read_pickle("pkl/reference_table_empty_brush.pkl")
master_empty = master_empty.loc[(master_empty.s == s) & (master_empty.r== r) & (master_empty.sigma == sigma)]
master = master.loc[master.chi_PC.isin(chi_PC)]
master = master.loc[master.ph.isin(ph)]
master = master.loc[master.chi_PS.isin(chi_PS)]
master = master.loc[master.sigma == sigma]
#%%
X = [0.7, -0.3]
#X = [1, 0]
gamma_f = gamma
if X is None:
    X0 = [1,0]
    fit_data = pd.read_pickle("pkl/reference_table.pkl")
    fit_data = fit_data.loc[(fit_data.ph == 4)&(fit_data.s == s)&(fit_data.r== r)]
    f = create_cost_function(fit_data, master_empty, gamma_f)
    from scipy.optimize import least_squares
    res = least_squares(f, X0)
    X =res.x
#%%
CHI_PC = master.chi_PC.unique()
CHI_PS = master.chi_PS.unique()
fig, axs = plt.subplots(
    nrows=len(CHI_PC)+1,
    ncols=len(CHI_PS),
    sharey = "row",
    sharex= True)

for chi_PS, ax in zip(CHI_PS, axs[0].flatten()):
    empty_pore_data = utils.get_by_kwargs(master_empty, chi_PS = chi_PS)
    phi_0 = empty_pore_data.dataset["phi"].squeeze()[0, :]
    x = list(range(-len(phi_0)//2, len(phi_0)//2))
    ax.plot(x, phi_0, color = "black")
    ax.set_ylim(0, 0.4)
    trans = transforms.blended_transform_factory(
    ax.transData, ax.transAxes
    )

    rect = mpatches.Rectangle((-s/2, 0), width=s, height=1, transform=trans,
                          color='lightgreen', alpha=0.1)
    ax.add_patch(rect)

first_it = True
for (chi_PC, chi_PS), ax in zip(itertools.product(CHI_PC, CHI_PS), axs[1:].flatten()):
    if first_it:
        sc = []
    for ph_ in ph:
        ax.plot([],[])
        print(chi_PC, chi_PS, ph_)
        empty_pore_data = utils.get_by_kwargs(master_empty, chi_PS = chi_PS)
        osm, sur = free_energy_cylinder(int(ph_/2), empty_pore_data, chi_PS, chi_PC, gamma_f, X)
        tot = osm+sur
        x = list(range(-len(tot)//2, len(tot)//2))
        ax.plot(x, tot, 
            #color = "red"
            )

        data = master.query(f"chi_PS == {chi_PS} & chi_PC == {chi_PC} & ph == {ph_}")
        ax.scatter(data["pc"], data["free_energy"], 
                marker = "s", 
                #fc = "none", 
                #ec = "red", 
                fc = ax.lines[-1].get_color(),
                s = 10, 
                linewidth = 0.5
                )
        sc_ = ax.scatter(-data["pc"], data["free_energy"], 
                marker = "s", 
                #fc = "none", 
                #ec = "red", 
                label = ph_,
                fc = ax.lines[-1].get_color(),
                s = 10, 
                linewidth = 0.5
                )
        if first_it: sc.append(sc_)
    first_it=False

    # ax.plot(x, osm, color = "darkorange", linewidth = 0.5, linestyle = "-")
    # ax.plot(x, sur, color = "blue", linewidth = 0.5, linestyle = "-")

    #osm, sur = free_energy_cylinder(int(pw/2), empty_pore_data, chi_PS, chi_PC, gamma_f, X, trunc =True)
    #tot = osm+sur
    #ax.plot(x, tot, color = "darkred", linestyle = "--")
    
    ax.set_xlim(-75, 75)
    #ax.set_ylim(-5, 8)

    
    #ax.axvline(-s/2, color = "grey", linestyle = "--", linewidth =0.5)
    #ax.axvline(s/2, color = "grey", linestyle = "--", linewidth =0.5)
    trans = transforms.blended_transform_factory(
    ax.transData, ax.transAxes
    )

    rect = mpatches.Rectangle((-s/2, 0), width=s, height=1, transform=trans,
                          color='lightgreen', alpha=0.1)
    ax.add_patch(rect)
    
    ax.axhline(0, color = "black", linewidth = 0.5)
    #ax.set_title(f"$\chi_{{PS}} = {chi_PS}$ $\chi_{{PC}} = {chi_PC}$")
    #ax.set_title("$\chi_{}$")
    #ax.set_xlabel("")
    #ax.set_ylabel("")



#Dummy plots for legend
lineplot, = ax.plot([], [], color = "red", label = r"$\Delta F_{\text{cyl}}$")
scatterplot = ax.scatter([], [], marker = "s", fc = "black", label = r"$\Delta F_{\text{SF-SCF}}$", s = 10)
#ax.plot([], [], color = "black", label = "$\phi(z)_{r=0}$")
#ax.plot([], [],  color = "darkorange", linewidth = 0.5, linestyle = "-", label = "$\Delta F_{osm}$")
#ax.plot([], [],  color = "blue", linewidth = 0.5, linestyle = "-", label = "$\Delta F_{sur}$")
#ax.plot([], [],color = "darkred", linestyle = "--", label = "$P>0$")

#Layout
axs[0,0].set_ylabel("$\phi$")
[ax_.set_ylabel("$\Delta F / k_B T$") for ax_ in axs[1:, 0]]
[ax_.set_xlabel("$z$") for ax_ in axs[-1, :]]
[ax_.set_title(f"$\chi_{{PS}} = {chi_PS_}$") for ax_, chi_PS_ in zip(axs[0, :], CHI_PS)]
[ax_.spines[['right', 'top']].set_visible(False) for ax_ in  axs.flatten()]

def add_text_right(ax_, text):
    ax_.text(1.1, 0.5, text,
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        transform=ax_.transAxes)

[add_text_right(ax_, f"$\chi_{{PC}} = {chi_PC_}$") for ax_, chi_PC_ in zip(axs[1:, -1], CHI_PC)]


#axs[-1, -1].legend(ncols = 5, bbox_to_anchor = [1.2, -.5])
first_legend = fig.legend(handles=[lineplot,scatterplot], title='calculation', bbox_to_anchor = [1.05, 0.6], loc = "upper right")
second_legend = fig.legend(handles=sc, title='particle size', bbox_to_anchor = [1.05, 0.5], loc = "upper right")

fig.set_size_inches(10,10)
#fig.savefig("/home/ml/Desktop/grid.svg")
# %%

# %%
# %%