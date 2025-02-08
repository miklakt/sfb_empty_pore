#%%
import itertools
import tqdm
import matplotlib
import matplotlib.colors as plt_colors
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.style as style
from matplotlib import rc
style.use('tableau-colorblind10')
mpl_markers = ('o', '+', 'x', 's', 'D')
rc('hatch', color='darkgreen', linewidth=9)
import cmasher as cmr

import pickle
import numpy as np
import h5py
import sys
import pathlib
here = pathlib.Path(os.path.dirname(__file__))
sys.path.append(os.path.join(here, '..'))
from heatmap_explorer import plot_heatmap_and_profiles


def load_c_arr_w_arr(chi_PC, chi, d=12,
            dt = 0.2,
            differencing = "power_law",
            zlayers = 392+100,
            rlayers = 66+200,
            wall_thickness = 52,
            pore_radius = 26,):
    simulation_name = \
        f"simulation_data/{d=}_{zlayers=}_{rlayers=}_{chi=}_{chi_PC=}_{dt=}_{differencing}.h5"
    simulation_results = h5py.File(simulation_name, "r")
    c_arr = simulation_results["c_arr"][-1].T
    W_arr = np.array(simulation_results["W_arr"], dtype = "bool").T
    c_arr[W_arr == True] = np.nan
    simulation_results.close()
    return c_arr, W_arr
#%%
chi_PC = -1.25
chi = 0.3
d=12

c_arr_d = {}
W_arr_d = {}

c_arr_d[chi_PC], W_arr_d[chi_PC]= load_c_arr_w_arr(chi_PC, chi, d)

chi_PC = -1.5
c_arr_d[chi_PC], W_arr_d[chi_PC]= load_c_arr_w_arr(chi_PC, chi, d)

chi_PC = -1.0
c_arr_d[chi_PC], W_arr_d[chi_PC]= load_c_arr_w_arr(chi_PC, chi, d)

chi_PC = -0.75
c_arr_d[chi_PC], W_arr_d[chi_PC]= load_c_arr_w_arr(chi_PC, chi, d)

chi_PC = 0
c_arr_d[chi_PC], W_arr_d[chi_PC]= load_c_arr_w_arr(chi_PC, chi, d)

#%%

c_arr_0 = np.loadtxt("simulation_data/empty_492_266_26_52_12.txt")
#%%

#%%
fig, ax = plt.subplots(dpi = 600)
# ax.set_xlim(-150, 150)
# ax.set_ylim(0, 150)
ax.set_xlim(-75, 75)
ax.set_ylim(0, 75)

#c_arr[W_arr == True] = np.nan

chi_PC = 0
c_arr = c_arr_d[chi_PC]
W_arr = W_arr_d[chi_PC]

cmap0_1 = cmr.get_sub_cmap("CMRmap_r", 0.0, 0.5)
cmap1_max = cmr.get_sub_cmap("CMRmap_r", 0.5, 1.0)


cmap0_1.set_bad(color='green')
cmap1_max.set_bad(color='green')

extent = [-zlayers/2, zlayers/2, 0, rlayers]
width  = extent[1] - extent[0]
height = extent[3] - extent[2]

bg_rect = matplotlib.patches.Rectangle(
    (extent[0], extent[2]),  # bottom-left corner
    width,                    # width
    height,                   # height
    facecolor="green",        # fill color
    hatch="/",             # the hatching pattern
    edgecolor=None,
    zorder=0                 # so it stays behind the imshow
)
ax.add_patch(bg_rect)


c_arr_im = ax.imshow(
    c_arr,
    cmap=cmap0_1, 
    extent=extent, 
    origin = "lower",  
    aspect = 'equal',
    vmin = 0.0,
    vmax = 1.0,
    alpha = (c_arr<=1).astype(float)
    #norm = norm_,
    )

gamma_=0.4
norm_ = plt_colors.PowerNorm(
    gamma=gamma_, 
    vmin=1.0, 
    #vmax=np.nanmax(c_arr)
    vmax = 40
    ) 
c_arr_im2 = ax.imshow(
    c_arr,
    cmap=cmap1_max, 
    extent=extent, 
    origin = "lower",  
    aspect = 'equal',
    alpha = (c_arr>1).astype(float),
    norm = norm_,
    interpolation = "none"
    )

ax.set_axis_off()

fig.savefig(here.parent/f"fig/streamlines/streamlines_contours_min_{chi_PC}.png", dpi =1200, transparent = True,
          bbox_inches='tight', pad_inches=0)
#%%


# levels = np.concatenate([np.arange(0.90, 1.0, 0.01), np.arange(0.0, 0.2, 0.01)])
# levels.sort()
# contour = ax.contour(
#     c_arr, 
#     extent = extent, 
#     colors = "black",
#     linewidths = 0.1,
#     levels = levels
#     )
# ax.clabel(contour, inline = False)


# x = np.arange(0, zlayers, 10)
# y = np.arange(0, rlayers, 10)
# xx, yy = np.meshgrid(x, y)
# J_arr = drift_diffusion.J_arr.get()
# uv = [J_arr[xx_, yy_] for xx_, yy_ in zip(xx, yy)]
# u = np.moveaxis(uv, -1, 0)[0]
# v = np.moveaxis(uv, -1, 0)[1]
# #norm = np.linalg.norm(np.array((u, v)), axis=0)

# xx = xx - zlayers/2

# ax.quiver(
#    xx, yy, u/norm*2, v/norm*2, 
#    width = 0.003,
#     headlength = 3,

#    color = 'grey'
#    )

# start_points_y = np.arange(1, 27,2)
# start_points_x = np.ones_like(start_points_y)*zlayers/2
# start_points_y = np.arange(0, rlayers-1,15)
# start_points_x = np.ones_like(start_points_y)-zlayers/2

# start_points = np.array([start_points_x, start_points_y]).T
# J_arr_stream = ax.streamplot(

#     xx, yy, u, v, 
#     #color = norm,
#     color = "grey",
#     start_points = start_points,
#     #broken_streamlines = False,
#     arrowsize = 0,
#     linewidth = 0.3,
#     density = 35
#     )

# c_arr_cbar = plt.colorbar(c_arr_im)
# c_arr_cbar2 = plt.colorbar(c_arr_im2)

# ax.set_xlabel("$z$")
# ax.set_ylabel("$r$")
# c_arr_cbar.set_label("$c/c_0$")

#fig.savefig("fig/streamlines/streamlines_contours.svg", dpi =1200, transparent = True)
#fig.savefig("fig/streamlines/streamlines_heatmap.png", dpi =600, transparent = True)

# %%
fig2, (ax2_upper, ax2_lower) = plt.subplots(
    2, 
    1, 
    sharex=True,                 # same x-axis for both subplots
    gridspec_kw={"height_ratios": [1, 2.5]},  # 2/3 height for lower, 1/3 for upper
    figsize=(4, 2.5),
)
fig2.subplots_adjust(hspace=0)
y_broke = 1
#max_val = np.nanmax(slice_data)
max_val=40
# --- Upper subplot: log scale from 1 to max_val ---
ax2_upper.set_yscale("log")
ax2_upper.set_ylim(y_broke, max_val**1.05)

ax2_lower.set_xlim(-75,75)

markers = itertools.cycle(mpl_markers)
for chi_pc in [-1.5, -1.25, -1.0, -0.75, 0]:
    marker = next(markers)
    y_index = 0
    slice_data = np.array(c_arr_d[chi_pc][y_index, :])
    x_vals = np.linspace(-zlayers/2, zlayers/2, zlayers)
    ax2_lower.plot(x_vals, slice_data, 
                    lw=1, marker = marker,
                    label=r'$\chi_{\text{PC}}='+f'{chi_pc}$',
                    markevery=10
                    )
    slice_data[slice_data<=y_broke]=np.nan
    ax2_upper.plot(x_vals, slice_data, 
                    lw=1, marker = marker,
                    markevery=10,
                    )



ax2_lower.plot(x_vals,c_arr_0[:,y_index], color = "k", lw=2, label=r'$c_{\text{empty}}$')

# --- Lower subplot: linear scale from 0 to 1 ---
ax2_lower.set_ylim(0, y_broke)


# Remove spines and ticks where the break happens
ax2_lower.spines["top"].set_visible(False)
ax2_upper.spines["bottom"].set_visible(False)
ax2_upper.xaxis.tick_top()
#ax2_upper.xaxis.set_ticks([])      
#ax2_lower.xaxis.tick_bottom()

# Add diagonal "break" marks
d = 0.015  # size of diagonal lines in axis coordinates
kwargs = dict(color='k', clip_on=False)

# Diagonals for lower subplot (top boundary)
ax2_lower.plot((-d, +d), (1 - d, 1 + d), transform=ax2_lower.transAxes, **kwargs)
ax2_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax2_lower.transAxes, **kwargs)

# Diagonals for upper subplot (bottom boundary)
ax2_upper.plot((-d, +d), (-d, +d), transform=ax2_upper.transAxes, **kwargs)
ax2_upper.plot((1 - d, 1 + d), (-d, +d), transform=ax2_upper.transAxes, **kwargs)

# Labels and legends
ax2_lower.set_xlabel("$r$")
ax2_lower.set_ylabel("$c(r=0)/c_0$")
ax2_lower.legend(loc="upper right")

plt.tight_layout()
plt.show()
fig2.savefig(here.parent/"fig/colloid_conc_stationary.svg")
# %%
