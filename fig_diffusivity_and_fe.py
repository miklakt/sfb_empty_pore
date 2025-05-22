#%%
# Pore resistance R, normalized by the viscosity of the solvent
# as a function of colloid size d 
# (a) for selected polymer-colloid interaction
# strengths at a fixed solvent strength
# (b) for selected solvent strengths (χPS, as indicated with colored lines) at a
# fixed polymer-colloid interaction strengths χPC = −1.25.
# The normalized resistance of an empty pore R0 
# (black thick lines) serves as a reference.
#%%
import itertools
import functools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rc, rcParams
import matplotlib.patches as mpatches
import matplotlib

rc('hatch', color='darkgreen', linewidth=9)

rcParams.update({
    "mathtext.fontset": "cm",  # Use Computer Modern
    "font.family": "serif",
})

style.use('tableau-colorblind10')
get_palette_colors = lambda: itertools.chain([
 '#FF800E',
 '#006BA4',
 '#ABABAB',
 '#595959',
 '#5F9ED1',
 '#C85200',
 '#898989',
 '#A2C8EC',
 '#FFBC79',
 '#CFCFCF'])
mpl_markers = ('+','o', '^', 's', 'D', 'd')

import calculate_fields_in_pore


def free_energy_gel(phi, chi_PS, chi_PC, d):
    from calculate_fields_in_pore import volume, surface, Pi, gamma
    V = volume(d)
    S = surface(d)
    Pi_ = Pi(phi, chi_PS, trunc=False)
    gamma_ = gamma(chi_PS, chi_PC, phi, a0, a1)
    free_energy = Pi_*V + gamma_*S
    #print(Pi_)
    # free_energy = gamma_*S
    return free_energy

a0 = 0.7
a1 = -0.3
L = 52
r_pore=26
sigma = 0.02
alpha =  30**(1/2)
d = 8
chi_PS = 0.5
chi_PCs =  [-0.5, -0.75, -1.0, -1.25]
#%%
# calculate_fields = functools.partial(
#     calculate_fields_in_pore.calculate_fields,
#         a0 = a0, a1 = a1, 
#         wall_thickness = L, 
#         pore_radius = r_pore,
#         sigma = sigma,
#         mobility_model_kwargs = {"prefactor":alpha},
#         chi_PS = chi_PS
#     )
calculate_fields = functools.partial(calculate_fields_in_pore.calculate_fields,
    a0=a0, a1=a1, 
    chi_PS=chi_PS,
    wall_thickness=L, 
    pore_radius=r_pore, d=d, 
    sigma=sigma,
    mobility_model_kwargs = {"prefactor": alpha}
    )

def plot_heatmap(ax,fields, r_cut, z_cut, key, vmin, vmax, cmap):
    from heatmap_explorer import plot_heatmap_and_profiles
    wall_thickness = fields["s"]
    l1 = fields["l1"]
    xlayers = fields["xlayers"]
    ylayers = fields["ylayers"]
    pore_radius = fields["r"]
    def cut_and_mirror(arr):
        cut = arr.T[0:r_cut, int(ylayers/2)-z_cut:int(ylayers/2)+z_cut]
        return np.vstack((np.flip(cut), cut[:,::-1]))
    extent = [-z_cut, z_cut, -r_cut, r_cut]
    mask = cut_and_mirror(fields["walls"]).T
    array = cut_and_mirror(fields[key]).T
    array = np.ma.array(array, mask = mask)
    cmap_ = matplotlib.cm.get_cmap(cmap)
    cmap_.set_bad(color="none")
    im = ax.imshow(
        array.T, 
        cmap=cmap_, 
        interpolation='nearest', 
        origin = "lower",
        extent = extent,
        vmin = vmin,
        vmax = vmax,
        )

    p = mpatches.Rectangle(
        (-wall_thickness/2, -r_cut), 
        wall_thickness, r_cut-pore_radius, 
        facecolor = "k", 
        edgecolor = "none", 
        #hatch ='/'
        alpha=0.1,
        )
    ax.add_patch(p)
    p = mpatches.Rectangle(
        (-wall_thickness/2, r_cut), 
        wall_thickness, -r_cut+pore_radius, 
        facecolor = "k", 
        edgecolor = "none", 
        #hatch ='/'
        alpha=0.1,
        )
    ax.add_patch(p)

    bg = mpatches.Rectangle(
        (0, 0), 1, 1,               # (x, y), width, height in axes coordinates
        transform=ax.transAxes,    # makes it relative to axes (0-1 range)
        facecolor='green',          # transparent fill
        edgecolor='darkgreen',         # hatch color
        hatch='/',               # hatch pattern
        zorder=-10                 # draw below everything else
    )
    ax.add_patch(bg)
    return im
# %%
# # Sample data
# x = np.linspace(-3, 3, 100)
# y = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(x, y)
# Z = np.sin(X**2 + Y**2)

# import pylustrator
# pylustrator.start()

fig = plt.figure(figsize=(7, 4.5))
subfigs = fig.subfigures(1, 2, width_ratios=[0.69, 2])
axs = subfigs[1].subplots(2, 2, 
                        #figsize=(3.5, 3.5), 
                        #constrained_layout=True,
                        #gridspec_kw={'wspace': 0, 'hspace': 0}
                        )
subfigs[1].subplots_adjust(left=0, right=1, top=1, bottom=0.05, wspace=-0.47, hspace=0)
subfigs[0].subplots_adjust(left=0, right=1, top=1, bottom=0.05, wspace=-0.33, hspace=0)

r_cut = 50
z_cut = 50
for ax, chi_PC in zip(axs.flatten(), chi_PCs):
    im = plot_heatmap(
        ax,
        calculate_fields(chi_PC = chi_PC),
        r_cut, z_cut,
        "free_energy",
        vmin = -5, vmax = 5,
        cmap = "seismic"
    )
    ax.text(0.02, 0.98, r"$\chi_{\text{PC}} = "+f"{chi_PC}$", 
        transform = ax.transAxes, va = "top", ha = "left",
        bbox ={"fc" : "white", "pad":1},
        fontsize = 12,
    )
    # ax.set_xticks([])
    # ax.set_yticks([])

for i in range(2):
    for j in range(2):
        ax = axs[i, j]
        if j != 0:
            ax.set_yticks([])
        else:
            ax.yaxis.set_ticks_position('left')
            ax.yaxis.set_tick_params(labelright=False)
            #ax.set_ylabel("$r$", fontsize=12, labelpad = -0.5)
        if i != 1:
            ax.set_xticks([])
        else:
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_tick_params(labeltop=False)
            ax.set_xticks([-40, -20, 0, 20, 40]),
            #ax.set_xlabel("$z$", fontsize =12, labelpad = -.5)

# # Create a single colorbar below all plots
cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1, shrink = 0.7)
cbar.set_label('$\Delta F$', fontsize = 12)


axs_left = subfigs[0].subplots(nrows=2, ncols = 1, height_ratios = [0.6, 1])
fields = calculate_fields(chi_PC = chi_PC)
fields["ln_mobility"] = -np.log(fields["mobility"])
im = plot_heatmap(
        axs_left[1],
        fields,
        r_cut, z_cut,
        "ln_mobility",
        vmin = 0, vmax = 5,
        cmap = "Reds"
    )
axs_left[1].set_xticks([-40, -20, 0, 20, 40])
#axs_left[1].set_xlabel("$z$", fontsize =12, labelpad = -.5)
#axs_left[1].set_ylabel("$r$", fontsize=12, labelpad = -.5)


phi = np.linspace(0,0.3)
mob = -np.log(calculate_fields_in_pore.mobility_Rubinstein(phi=phi, d=d, prefactor=alpha, k=1))
axs_left[0].plot(phi, mob, color = "k", linewidth = 2)
colors =  itertools.chain(["#001483","#003F91",'#006BA4',"#FF6A6A", "#DB1919"][::-1])
for chi_PC in chi_PCs+[-1.5]:
    fe = free_energy_gel(phi, chi_PS, chi_PC, d)
    color_ =next(colors)
    axs_left[0].plot(phi, fe, color =color_),
    axs_left[0].text(phi[-25], fe[-25], chi_PC,  color =color_,
                     bbox ={"fc" : "white", "ec":"none", "pad":0.2},)

axs_left[0].axhline(0, color = "k", linewidth = 0.2)
axs_left[0].set_xlabel("$\phi$", fontsize =12, labelpad = -.2)
axs_left[0].set_ylabel(r"$-\ln(D/D_0),\, \Delta F$", fontsize=12)
axs_left[0].set_xlim(0,0.3)
axs_left[0].set_ylim(-5.8,5.8)

cbar = fig.colorbar(im, ax=axs_left, orientation='horizontal', fraction=.05, pad=0.1, shrink = 1.0, )
cbar.set_label('$-\ln(D/D_0)$', fontsize = 12)

# Grab the colorbar Axes
cbar_ax = cbar.ax
pos = cbar_ax.get_position()

# Replace it with a new Axes that is taller, same center position
extra = 0.02  # How much taller to make it (in figure-relative units)
new_height = pos.height + extra
new_y = pos.y0 - extra / 2  # shift downward to preserve center

# Set the new position
cbar_ax.set_position([pos.x0, new_y, pos.width, new_height])
fig.savefig("fig/diffusivity_and_fe.svg")
#plt.show()
# %%
