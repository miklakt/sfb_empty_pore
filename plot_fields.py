#%%
from calculate_fields_in_pore import *
from heatmap_explorer import plot_heatmap_and_profiles
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sfbox_utils
#%%
def plot_heatmap(fields, r_cut, z_cut, keys, **kwargs):
    from heatmap_explorer import plot_heatmap_and_profiles
    wall_thickness = fields["s"]
    l1 = fields["l1"]
    def cut_and_mirror(arr):
        cut = arr.T[0:r_cut, l1-z_cut:l1+wall_thickness+z_cut]
        return np.vstack((np.flip(cut), cut[:,::-1]))[:,::-1]
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
            hline_y=r_cut,
            vline_x=int(z_cut+wall_thickness/2),
            mask = mask.T,
            **kwargs
            )
        fig.show()
        #fig.savefig(f"kernel_figs/{key}.pdf")
    return fig

#%%
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness
# %%
import matplotlib.colors as plt_colors
a0, a1 = 0.7, -0.3
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness
d = 8
chi_PC = -1.4
chi_PS = 0.5
sigma = 0.02

kwargs = dict(
    a0=a0, a1=a1, d=d,
    chi_PC=chi_PC, chi_PS=chi_PS,
    sigma = sigma,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    mobility_model_kwargs = {"prefactor":30.0**(1/2)},
    stickiness=False,
    #gel_phi = 0.2
)
fields = calculate_fields(
    **kwargs
    )

fields["resistivity"] = (fields["conductivity"])**(-1)
fields["pc"] = np.exp(-fields["free_energy"])
fields["mobility"] = -np.log(fields["mobility"])
#fields["c"] = fields["psi"]*np.exp(-fields["free_energy"])
#%%
%matplotlib TkAgg
import cmasher as cmr
#cmap = cmr.
cmap0 = cmr.get_sub_cmap("seismic", 0.0, 0.5)
cmap1 = cmr.get_sub_cmap("seismic", 0.5, 1.0)
vmin, vmax = -2, 2
#cmap_ = cmr.combine_cmaps(cmap0, cmap1, nodes=[(1-vmin)/(vmax-vmin)])
r_cut = 50
z_cut = 40
fig = plot_heatmap(fields, r_cut, z_cut, keys = [
    #"phi", 
    #"Pi", 
    #"gamma", 
    "c", 
    #"mobility", 
    #"conductivity", 
    #"resistivity",
    #"osmotic", 
    #"surface"
    #"pc"
    ], 
    #cmap = "Reds",
    cmap = "CMRmap_r",
    #zmin=0,
    #zmax = 4,
    #cmap = cmap_.reversed(),
    #cmap = cmap_,
    #zmin=vmin,
    #zmax = vmax,

    )
#fig.savefig(f"fig/free_energy/free_energy_{chi=}_{chi_PC=}_{d=}.svg")
#fig.savefig(f"fig/free_energy/resistivity_{chi=}_{chi_PC=}_{d=}.svg")
# %%
