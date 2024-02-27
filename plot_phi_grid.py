#%%
import matplotlib
import matplotlib.pyplot as plt
from utils import get_by_kwargs
import pandas as pd
import numpy as np
import matplotlib.colors as plt_colors

import itertools
import matplotlib.style as style
style.use('tableau-colorblind10')
mpl_markers = ('o', '+', 'x', 's', 'D')


def mask_field(field, key ="phi", r_cut = None, z_cut = None, mirror = True):
    arr = field.dataset[key].squeeze()
    wall_thickness = field["s"].squeeze()
    pore_radius = field["r"].squeeze()
    l1 = field["l1"].squeeze()
    W_arr = np.zeros_like(arr)
    W_arr[pore_radius:, l1:l1+wall_thickness] = True
    z_center = l1+wall_thickness/2
    if r_cut is None:
        r_cut = np.shape(arr)[0]
    if z_cut is None:
        z_cut = np.shape(arr)[1]/2
    lz = int(z_center-z_cut)
    rz = int(z_center+z_cut)
    
    arr_masked = np.ma.array(arr, mask = W_arr)
    arr_masked = arr_masked[0:r_cut, lz:rz]
    
    if mirror:
        arr_masked = np.ma.vstack((np.flip(arr_masked), arr_masked[:,::-1]))
        extent = [-z_cut, +z_cut, -r_cut, r_cut]
    else:
        extent = [-z_cut, +z_cut, 0, r_cut]
    return arr_masked, extent


r_cut = None
z_cut = 80
fields = pd.read_pickle("reference_table_empty_brush.pkl")
#fields = get_by_kwargs(fields, r = 26, s = 52)
fields = get_by_kwargs(fields, chi_PS = 0.7).iloc[1]

CHI_PS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
CHI_PS = [0.7]

#fig, axs = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True)
fig, axs = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True)

vmin = 0
vmax = 0.75
#for chi_ps, ax in zip(CHI_PS, axs.flatten()):
for chi_ps, ax in zip(CHI_PS, [axs]): 
    #field = get_by_kwargs(fields, chi_PS = chi_ps)
    field = fields
    phi_masked, extent = mask_field(field,"phi", r_cut, z_cut)
    #cmap_ = matplotlib.cm.get_cmap("gist_stern_r")
    cmap_ = matplotlib.cm.get_cmap("gnuplot2_r")
    cmap_.set_bad(color='green')
    gamma_=0.4
    norm_ = plt_colors.PowerNorm(gamma=gamma_, vmin=vmin, vmax=vmax) 
    im = ax.imshow(
        phi_masked, 
        cmap=cmap_, 
        extent=extent, 
        origin = "lower",  
        aspect = 'equal',
        #vmin = vmin,
        #vmax = vmax,
        norm = norm_,
        )
    if chi_ps<1.6:
        levels = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6]
        contour = ax.contour(
            phi_masked, 
            extent = extent, 
            colors = "white",
            linewidths = 0.3,
            levels = levels
            )
        #ax.clabel(contour)

fig.supxlabel("$z$")
fig.supylabel("$r$")

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes((1.0, 0.2, 0.04, 0.6))
# cbar = plt.colorbar(
#     im, cax = cax, 
#     #extend='max'
#     )
# cbar.add_lines(contour)

cax.imshow(np.array([np.linspace(0,vmax, 600)]*1).T,
           norm = norm_,
           cmap = cmap_,
           origin = "lower",
           extent = [0, 0.07, 0, 0.7],
           aspect = "auto"
           )
[cax.axhline(level, color = "white", linewidth = 0.7) for level in levels]
cax.yaxis.tick_right()
cax.set_xticks([], minor = False)
cax.set_title("$\phi$")
plt.tight_layout()
fig.set_size_inches(2.5, 3)

#fig.savefig("/home/ml/Desktop/phi_open.svg")
#%%
fields = pd.read_pickle("reference_table_empty_brush.pkl")
fields = get_by_kwargs(fields, chi_PS = 0.7).iloc[1]

#%%
r_cut = None
z_cut = 80

# %%
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
fig, ax = plt.subplots()
markers = itertools.cycle(mpl_markers)
trans = transforms.blended_transform_factory(
        ax.transData, ax.transAxes
        )

for chi_ps in CHI_PS:
    field = get_by_kwargs(fields, chi_PS = chi_ps)
    wall_thickness = field["s"].squeeze()
    pore_radius = field["r"].squeeze()
    l1 = field["l1"].squeeze()
    phi = field.dataset["phi"].squeeze()
    phi_r0 = phi[0,:]

    z = np.arange(len(phi_r0)) - len(phi_r0)/2

    ax.plot(z, phi_r0, 
            label = chi_ps,
            #marker = next(markers),
            #markerfacecolor = "None",
            #markevery = 0.2,
            )
    
rect = mpatches.Rectangle((-wall_thickness/2, 0), width=wall_thickness, height=1, transform=trans,
                        color='grey', alpha=0.1)
ax.add_patch(rect)
ax.legend(title="$\chi_{PS}$")
ax.set_xlim(-80,80)
ax.set_ylim(-0.02, 0.8)

ax.set_xlabel("$z$")
ax.set_ylabel("$\phi(z, r=0)$")

fig.set_size_inches(3.5, 3)

fig.savefig("/home/ml/Desktop/phi_center.svg")
# %%
