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
    return fig

#%%
# %%
a0, a1 = 0.70585835, -0.31406453
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness
d = 8
chi_PC = -1.25
chi = 0.5
sigma = 0.02

fields = calculate_fields(
    a0, a1, d=d,
    chi_PC=chi_PC, chi=chi,
    sigma = sigma,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    exclude_volume=True,
    truncate_pressure=False,
    method= "convolve", 
    mobility_correction= "vol_average",
    mobility_model = "Rubinstein",
    mobility_model_kwargs = {"prefactor":1.0}
    #**method
    )

perm = calculate_permeability(    
    a0=a0, a1=a1, d=d,
    chi_PC=chi_PC, chi_PS=chi,
    sigma = sigma,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    exclude_volume=True,
    truncate_pressure=False,
    method= "convolve",
    convolve_mode="same",
    mobility_correction= "vol_average",
    mobility_model = "Rubinstein",
    mobility_model_kwargs = {"prefactor":1.0},         
    integration= "cylindrical_caps",
    integration_kwargs = dict(spheroid_correction = True)
    )

fields["resistivity"] = np.log10(fields["conductivity"])

#import cmasher as cmr
#cmap0 = cmr.get_sub_cmap("seismic", 0.0, 0.5)
#cmap1 = cmr.get_sub_cmap("seismic", 0.5, 1.0)
#vmin, vmax = -2, 2
#cmap_ = cmr.combine_cmaps(cmap0, cmap1, nodes=[(1-vmin)/(vmax-vmin)])
r_cut = 50
z_cut = 30
fig = plot_heatmap(fields, r_cut, z_cut, keys = [
    #"phi", 
    #"Pi", 
    #"gamma", 
    "free_energy", 
    #"mobility", 
    #"conductivity", 
    #"osmotic", 
    #"surface"
    ], 
    cmap = "seismic",
    zmin=-4,
    zmax = 4,
    )
fig.savefig(f"fig/free_energy/free_energy__{chi=}_{chi_PC=}_{d=}.svg")
#%%
