#%%
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import itertools

import numpy as np
import utils
import pandas as pd
from scipy.signal import convolve
import sfbox_utils
import seaborn as sns
#%%
def cylynder_r0_kernel(radius:int, height:int = None):
    if height is None:
        height = radius*2
    r = np.arange(radius)
    volume_r = np.pi*(2*r+1)
    volume = np.tile(volume_r, (height,1)).T
    surface = np.zeros_like(volume)

    surface[:, 0] = surface[:,-1] = volume_r
    surface[-1,:] =surface[-1,:] + 2*np.pi*radius
    return volume, surface

def cylinder_volume_surface(radius:int, height:int = None):
    if height is None:
        height = radius*2
    V = np.pi*radius**2*height
    S = 2*np.pi*radius**2 + 2*np.pi*radius*height
    return V, S

def Pi(phi, chi_PS, trunc = False):
    Pi_=-np.log(1-phi) - phi - chi_PS*phi**2
    if trunc:
        Pi_[Pi_<1e-16]=0
    return Pi_

def gamma(chi_PS, chi_PC, phi, X):
    a0, a1 = X
    chi_crit = 6*np.log(5/6)
    phi_corrected = (a0 + a1*chi_PC)*phi
    chi_ads = chi_PC - chi_PS*(1-phi_corrected)
    #chi_ads = chi_PC - chi_PS*(1-phi)
    gamma = (chi_ads - chi_crit)*phi_corrected/6
    #gamma = (chi_ads - chi_crit)*phi/6
    return gamma

def gamma2(chi_PS, chi_PC, phi, X):
    a0, a1, a2 = X
    chi_crit = 6*np.log(5/6)
    #phi_corrected = (a0 + a1*chi_PC + a2*chi_PS)*phi
    chi_ads = a0*chi_PC - a1*chi_PS*chi_PC + a2*chi_PS
    gamma = (chi_ads - chi_crit)*phi_corrected/6
    return gamma

def free_energy_cylinder(radius, data, chi_PS, chi_PC, gamma_func, X_args, trunc = False):
    volume, surface = cylynder_r0_kernel(radius)
    phi = data.dataset["phi"].squeeze()
    if np.shape(phi)[0] == 1:
        phi = np.tile(phi, (radius, 1))
    phi = np.pad(phi[0:radius], ((0, 0),(radius,radius-1)))
    Pi_arr = Pi(phi, chi_PS, trunc)
    gamma_arr = gamma_func(chi_PS, chi_PC, phi, X_args)
    osmotic = convolve(Pi_arr, volume, 'valid')[0]
    surface = convolve(gamma_arr, surface, 'valid')[0]
    #extra = X_args[2]*radius**2
    return osmotic, surface#, extra

def free_energy_approx(radius, data, chi_PS, chi_PC, gamma_func, X_args, trunc = False):
    volume, surface =cylinder_volume_surface(radius)
    phi = data.dataset["phi"].squeeze()[0, :]
    Pi_arr = Pi(phi, chi_PS, trunc)
    gamma_arr = gamma_func(chi_PS, chi_PC, phi, X_args)
    osmotic = Pi_arr*volume
    surface = gamma_arr*surface
    return osmotic, surface

def create_cost_function(df, df_empty, gamma_func):
    def cost_function(X):
        cost = np.array([])
        for (chi_PS, chi_PC, pw), group in df.groupby(by = ["chi_PS", "chi_PC", "pw"]):
            empty_pore_data = utils.get_by_kwargs(df_empty, chi_PS = chi_PS)
            if empty_pore_data.empty:
                continue
            osm, sur = free_energy_cylinder(
                int(pw/2), empty_pore_data, 
                chi_PS, chi_PC, gamma_func, X,
                )
            tot = osm+sur
            delta_fe = group.apply(lambda _: _.free_energy - tot[int(_.pc+len(tot)//2)], axis = 1)
            delta_fe = np.cbrt(delta_fe)
            cost = np.concatenate([cost, delta_fe.to_numpy()])
        return cost
    return cost_function

X = None
#%%
s = 52
r = 26
ph =[8, 12, 16]
pw =ph
sigma = 0.02
chi_PS = [0.3, 0.4, 0.5, 0.6, 0.7]
chi_PC = [-1.5, 0.0]

master = pd.read_pickle("reference_table.pkl")
#master = master.loc[master["comment"] == "grown_from_small"]
master_empty = pd.read_pickle("reference_table_empty_brush.pkl")
master_empty = master_empty.loc[(master_empty.s == s) & (master_empty.r== r) & (master_empty.sigma == sigma)]
master = master.loc[master.chi_PC.isin(chi_PC)]
master = master.loc[master.ph.isin(ph)]
master = master.loc[master.chi_PS.isin(chi_PS)]
master = master.loc[master.sigma == sigma]
#%%
#X = [0.70585835, -0.31406453]
#[ 1.00029355e+00, -9.37511200e-04]
#X = [1, 0]
gamma_f = gamma
if X is None:
    X0 = [1,0]
    fit_data = pd.read_pickle("reference_table.pkl")
    fit_data = fit_data.loc[(fit_data.ph == 4)&(fit_data.s == s)&(fit_data.r== r)]
    f = create_cost_function(fit_data, master_empty, gamma_f)
    from scipy.optimize import least_squares
    res = least_squares(f, X0)
    X =res.x
#%%
#X = [1, 0]
#X = [1, 0]

# g = sns.FacetGrid(
#     master, 
#     row = "chi_PC", col = "chi_PS", 
#     hue = "ph", 
#     #sharey= "row", 
#     sharey = True,
#     sharex= True,
#     hue_kws=dict(marker = "s", color = "red")
#     )
# #g.map_dataframe(sns.scatterplot, x="pc", y="free_energy")
# g.set_axis_labels(x_var = "$z$", y_var = "$F / k_BT$")
#gamma_f = gamma
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
    ax.set_ylim(0, 0.3)
    trans = transforms.blended_transform_factory(
    ax.transData, ax.transAxes
    )

    rect = mpatches.Rectangle((-s/2, 0), width=s, height=1, transform=trans,
                          color='lightgreen', alpha=0.1)
    ax.add_patch(rect)

for (chi_PC, chi_PS), ax in zip(itertools.product(CHI_PC, CHI_PS), axs[1:].flatten()):
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
        ax.scatter(-data["pc"], data["free_energy"], 
                marker = "s", 
                #fc = "none", 
                #ec = "red", 
                fc = ax.lines[-1].get_color(),
                s = 10, 
                linewidth = 0.5
                )
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




#ax.plot([], [], color = "black", label = "$\phi(z)_{r=0}$")
ax.plot([], [], color = "red", label = "$\Delta F_{SS}$")
#ax.plot([], [],  color = "darkorange", linewidth = 0.5, linestyle = "-", label = "$\Delta F_{osm}$")
#ax.plot([], [],  color = "blue", linewidth = 0.5, linestyle = "-", label = "$\Delta F_{sur}$")
ax.scatter([], [], marker = "s", fc = "black", label = "$\Delta F_{SF}$", s = 10)

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
axs[-1, -1].legend()
#ax.plot([], [],color = "darkred", linestyle = "--", label = "$P>0$")
fig.set_size_inches(10,10)
#plt.tight_layout()
#fig.savefig("/home/ml/Desktop/grid.svg")
#%%
#%%
volume, surface = cylynder_r0_kernel(8, 16)
fig, ax = plt.subplots(nrows = 2, sharex = True)
fig.set_size_inches(4,4.5)

ax[0].pcolormesh(volume, edgecolor = "black", linewidth = 0.7)
ax[0].axis("equal")

ax[1].pcolormesh(surface, edgecolor = "black", linewidth = 0.7)
ax[1].axis("equal")

ax[1].set_xticks(np.arange(0,18, 2))

fig.savefig("/home/ml/Desktop/cylindrical_kernel.svg")
#%%
s = 52
r = 26
chi_PS = 0.4

master = pd.read_pickle("reference_table.pkl")
master = master.loc[master["comment"] == "grown_from_small"]
master_empty = pd.read_pickle("reference_table_empty_brush.pkl")
master_empty = master_empty.loc[
    (master_empty.s == s) & (master_empty.r== r)]
master = master.loc[
    master.chi_PC.isin([-1.5, -1.0, -0.5, 0]) & (master.chi_PS==chi_PS)]

g = sns.FacetGrid(
    master, 
    col = "chi_PC", 
    row = "ph", 
    hue = "chi_PS", 
    sharey=False, 
    #sharey= True,
    hue_kws=dict(marker = "s")
    )
g.map_dataframe(sns.scatterplot, x="pc", y="free_energy")


for (ph, chi_PC), ax in g.axes_dict.items():
    empty_pore_data = utils.get_by_kwargs(master_empty, chi_PS = chi_PS)
    osm, sur = free_energy_cylinder(int(ph/2), empty_pore_data, chi_PS, chi_PC, gamma_f, X)
    tot = osm+sur
    x = list(range(-len(tot)//2, len(tot)//2))
    ax.plot(x, tot, color = "red")

    #osm, sur = free_energy_cylinder(int(ph/2), empty_pore_data, chi_PS, chi_PC, gamma_f, X, trunc =True)
    #tot = osm+sur
    #ax.plot(x, tot, color = "darkred", linestyle = "--")
ax.plot([], [], color = "red", label = "model")
ax.plot([], [],color = "darkred", linestyle = "--", label = "$P>0$")
ax.legend()
#%%
chi_crit = 6*np.log(5/6)
chi_pc_crit = (1-X[0])/X[1]

fig, ax = plt.subplots()

CHI_PC = np.linspace(-1.5, 0, 50)
F = CHI_PC*X[1] + X[0]

ax.plot(CHI_PC, F)
ax.axhline(1, color = "black")
#ax.axvline(chi_pc_crit, color = "black", linestyle = "--")
ax.text(chi_pc_crit, 1, f"$\chi^{{*}}_{{PC}} = {chi_pc_crit:.2f}$", va = "bottom")
ax.set_xlabel("$\chi_{PC}$")
ax.set_ylabel("$\phi^{*} / \phi$")
fig.set_size_inches(3,3)
plt.tight_layout()
#fig.savefig("/home/ml/Desktop/phi_correction.png", dpi =600)
#%%

# %%
ph =8
pw = ph
#X=[1, 0]
X = [0.70585835, -0.31406453]
master = pd.read_pickle("reference_table_planar.pkl")
master_empty = pd.read_pickle("reference_table_planar_empty.pkl")
master = master.loc[
    master.chi_PS.isin([0.3, 0.4, 0.5, 0.6, 0.7, 0.75]) & 
    (master.ph==ph) & 
    (master.pw==pw)& (master.sigma==0.02)
    ]

gamma_f = gamma

g = sns.FacetGrid(
    master, 
    row = "chi_PC", col = "chi_PS", 
    hue = "ph", 
    sharey=False, 
    hue_kws=dict(marker = "s")
    )
g.map_dataframe(sns.scatterplot, x="pc", y="free_energy")
g.add_legend()

for (chi_PC, chi_PS), ax in g.axes_dict.items():
    empty_pore_data = utils.get_by_kwargs(master_empty, chi_PS = chi_PS)
    osm, sur = free_energy_cylinder(int(pw/2), empty_pore_data, chi_PS, chi_PC, gamma_f, X)
    tot = osm+sur
    x = list(range(len(tot)))
    ax.plot(x, tot, color = "red")

    osm, sur = free_energy_cylinder(int(pw/2), empty_pore_data, chi_PS, chi_PC, gamma_f, X, trunc =True)
    tot = osm+sur
    ax.plot(x, tot, color = "darkred", linestyle = "--")
# %%
s = 52
r = 26
ph =4
pw =ph
sigma = 0.02
chi_PS = [0.5]
chi_PC = [-1.5, -0.75, 0.0]

master = pd.read_pickle("reference_table.pkl")
#master = master.loc[master["comment"] == "grown_from_small"]
master_empty = pd.read_pickle("reference_table_empty_brush.pkl")
master_empty = master_empty.loc[(master_empty.s == s) & (master_empty.r== r) & (master_empty.sigma == sigma)]
master = master.loc[master.chi_PC.isin(chi_PC)]
master = master.loc[master.ph==ph]
master = master.loc[master.chi_PS.isin(chi_PS)]
master = master.loc[master.sigma == sigma]
# %%
fig, ax = plt.subplots()
gamma_f = gamma
for chi_PC_, df in master.groupby(by ="chi_PC"):
    empty_pore_data = utils.get_by_kwargs(master_empty, chi_PS = chi_PS)
    X = [1, 0]
    osm, sur = free_energy_cylinder(int(ph/2), empty_pore_data, chi_PS, chi_PC_, gamma_f, X)
    tot = osm+sur
    tot = tot[:len(tot)//2]
    x = list(range(-len(tot), 0))
    ax.plot(x, tot, 
        linestyle = "--",
        linewidth= 0.7,
        )
    X = [0.70585835, -0.31406453]
    osm, sur = free_energy_cylinder(int(ph/2), empty_pore_data, chi_PS, chi_PC_, gamma_f, X)
    tot = osm+sur
    tot = tot[:len(tot)//2]
    x = list(range(-len(tot), 0))
    ax.plot(x, tot, 
        color = ax.lines[-1].get_color(),
        label = "$\chi_{PC}=$" + str(chi_PC_)
        )
    df = df.loc[df.pc<=0]
    ax.scatter(df.pc, df.free_energy, marker = "o", s =20, facecolor = "none", edgecolor = ax.lines[-1].get_color())

ax.scatter([],[], marker = "o", s =20, facecolor = "none", edgecolor = 'black', label = "SF-SCF")
ax.plot([],[], linestyle = "--", linewidth= 0.7, color = "k", label = "$b_0 = 1.0, b_1 = 0.0$")
ax.plot([],[], color = "k", label = "$b_0 = 0.7, b_1 = -0.3$")

ax.axvline(-26, color = "green")


ax.legend(title = r"")
ax.set_xlim(-60,0)

ax.set_xlabel("z")
ax.set_ylabel("$\Delta F / k_B T$")

fig.set_size_inches(4,3)
fig.savefig("fig/fit.svg")
# %%
