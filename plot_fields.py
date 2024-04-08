#%%
from calculate_fields_in_pore import *
from heatmap_explorer import plot_heatmap_and_profiles
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sfbox_utils
#%%
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
        #fig.savefig(f"kernel_figs/{key}.pdf")
    #return fig

#%%
# %%
a0, a1 = [0.70585835, -0.31406453]
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness

d = 8
chi_PC = -1.5
chi = 0.5
sigma = 0.03
fields = calculate_fields(
    a0, a1, d=d,
    chi_PC=chi_PC, chi=chi,
    sigma = sigma,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    #**method
    )
# %%
r_cut = 50
z_cut = 30
plot_heatmap(fields, r_cut, z_cut, keys = ["phi", "Pi", "gamma", "free_energy", "mobility", "conductivity"])
#%%
d = 8
chi_PC = [0, -0.5, -1.0, -1.5]
chi = 0.5
fields = [calculate_fields(
    a0, a1, d=d,
    chi_PC=_i, chi=chi,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    #**method
    ) for _i in chi_PC]

fig, ax = plt.subplots()
for chi_PC_, field_ in zip(chi_PC, fields):
    Z = np.arange(field_["ylayers"])
    Z = Z -  field_["ylayers"]/2
    ax.plot(Z,field_["free_energy"][:,0], label = chi_PC_)
ax.legend(title = "$\chi_{PC}$")
ax.set_xlabel("$z$")
ax.set_ylabel("$\Delta F(r=0)$")

ax.axvline(field_["s"]/2, color = "black", linestyle = "--")
ax.axvline(-field_["s"]/2, color = "black", linestyle = "--")

# %%
d = 8
chi_PC = -1.0
chi = [0.3, 0.4, 0.5, 0.6]
fields = [calculate_fields(
    a0, a1, d=d,
    chi_PC=chi_PC, chi=_i,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    #**method
    ) for _i in chi]
#%%
fig, ax = plt.subplots()
for chi_PS_, field_ in zip(chi, fields):
    Z = np.arange(field_["ylayers"])
    Z = Z -  field_["ylayers"]/2
    ax.plot(Z,field_["free_energy"][:,0], label = chi_PS_)
ax.legend(title = "$\chi_{PS}$")
ax.set_xlabel("$z$")
ax.set_ylabel("$\Delta F(r=0)$")

ax.axvline(field_["s"]/2, color = "black", linestyle = "--")
ax.axvline(-field_["s"]/2, color = "black", linestyle = "--")
# %%
d = [2, 4, 8, 16]
chi_PC = -1.0
chi = 0.5
fields = [calculate_fields(
    a0, a1, d=_i,
    chi_PC=chi_PC, chi=chi,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    #**method
    ) for _i in d]

fig, ax = plt.subplots()
for d_, field_ in zip(d, fields):
    Z = np.arange(field_["ylayers"])
    Z = Z -  field_["ylayers"]/2
    ax.plot(Z,field_["free_energy"][:,0], label = d_)
ax.legend(title = "$d$")
ax.set_xlabel("$z$")
ax.set_ylabel("$\Delta F(r=0)$")

ax.axvline(field_["s"]/2, color = "black", linestyle = "--")
ax.axvline(-field_["s"]/2, color = "black", linestyle = "--")
# %%
d = 8
chi_PC = -1.0
#chi = [0.1, 0.2, 0.3, 0.4]
#chi = [0.5, 0.6, 0.7, 0.8, 0.9]
chi = [0.1, 0.3, 0.5, 0.7, 0.9]
fields = [calculate_fields(
    a0, a1, d=d,
    chi_PC=chi_PC, chi=_i,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    #**method
    ) for _i in chi]
#%%
fig, ax = plt.subplots()
for chi_PS_, field_ in zip(chi, fields):
    Z = np.arange(field_["ylayers"])
    Z = Z -  field_["ylayers"]/2
    ax.plot(Z[40:-40],field_["phi"][40:-40,0], label = chi_PS_)
ax.legend(title = "$\chi_{PS}$")
ax.set_xlabel("$z$ [Kuhn segments]")
ax.set_ylabel("$\phi (r=0)$")

ax.axvline(field_["s"]/2, color = "black", linestyle = "--")
ax.axvline(-field_["s"]/2, color = "black", linestyle = "--")
fig.set_size_inches(3,3)
plt.tight_layout()
fig.savefig("/home/ml/Desktop/fig1.png", dpi =600)
# %%
from utils import get_by_kwargs
d = 8
chi_PC = -1.0
chi = 0.5
fields = calculate_fields(
    a0, a1, d=d,
    chi_PC=chi_PC, chi=chi,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    sigma = 0.02,

    )

#%%
dataset = pd.read_pickle("reference_table.pkl")
#dataset = dataset.query("comment == 'grown_from_small'")
dataset = get_by_kwargs(dataset, chi_PS = chi, ph = d, pw = d, chi_PC = chi_PC)
dataset.sort_values(by = "pc", inplace = True)
# %%
fig, axs = plt.subplots(nrows = 2)

ax = axs[1]
ax.plot(dataset["pc"], dataset["free_energy"], linewidth = 0, marker = "s")

ax = axs[0]
ax.imshow(dataset.iloc[0].dataset["phi"].squeeze(), origin = "lower")
# %%
