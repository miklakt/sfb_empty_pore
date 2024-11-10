#%%
import tqdm
import matplotlib
import matplotlib.colors as plt_colors
import matplotlib.pyplot as plt
import matplotlib.transforms
import cmasher as cmr
import pickle
import numpy as np
import h5py
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from heatmap_explorer import plot_heatmap_and_profiles
#%%
chi_PC = -1.5
chi = 0.3
zlayers = 392+100
rlayers = 66+200
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness
pore_radius = 26
wall_thickness = 52
d=12
dt = 0.2
differencing = "power_law"

#simulation_name = \
#    f"simulation_data/{d=}_{zlayers=}_{rlayers=}_{chi=}_{chi_PC=}_{dt=}_{differencing}.h5"

#simulation_name = \
#    f"simulation_data/{d=}_{zlayers=}_{rlayers=}_{chi=}_{chi_PC=}_{dt=}_{differencing}.h5"
simulation_name = ""
#%%
simulation_results = h5py.File(simulation_name, "r")
#%%
c_arr = simulation_results["c_arr"][-1].T
W_arr = np.array(simulation_results["W_arr"], dtype = "bool").T
#%%
fig, ax = plt.subplots()

c_arr[W_arr == True] = np.nan

cmap0_1 = cmr.get_sub_cmap("CMRmap_r", 0.0, 0.5)
cmap1_max = cmr.get_sub_cmap("CMRmap_r", 0.5, 1.0)

cmap0_1.set_bad(color='green')
cmap1_max.set_bad(color='green')

extent = [-zlayers/2, zlayers/2, 0, rlayers]


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
norm_ = plt_colors.PowerNorm(gamma=gamma_, vmin=1.0, vmax=np.nanmax(c_arr)) 
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
fig, ax = plt.subplots()
z_center = int(zlayers/2)
ax.plot(c_arr[:,z_center])
ax.plot(c_arr[:,z_center-12])
ax.plot(c_arr[:,z_center-24])
ax.plot(c_arr[:,z_center-36])
ax.plot(c_arr[:,z_center-48])
ax.plot(c_arr[:,z_center-60])

# ax.plot(c_arr[:,z_center+12])
# ax.plot(c_arr[:,z_center+24])
# ax.plot(c_arr[:,z_center+36])

ax.set_xlim(0,50)