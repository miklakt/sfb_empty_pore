#%%    
import calculate_fields_in_pore
from solve_poisson import PoissonSolver2DCylindrical, pad_fields
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import rc
import cmasher as cmr
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
rc('hatch', color='darkgreen', linewidth=9)
#%%
a0 = 0.7
a1 = -0.3
L = 52
r_pore = 26
sigma = 0.02
alpha = 30**(1/2)
d = 8
chi_PC = -1.4
chi_PS = 0.5

fields = calculate_fields_in_pore.calculate_fields(
    a0=a0, a1=a1, 
    chi_PC=chi_PC,
    chi_PS=chi_PS,
    wall_thickness = L, 
    pore_radius = r_pore,
    d=d,
    sigma = sigma,
    mobility_model_kwargs = {"prefactor":alpha},
    linalg=False,
    #gel_phi=0.3,
    #method="approx"
)

conductivity, source = pad_fields(fields, z_boundary=300)
poisson = PoissonSolver2DCylindrical(D=conductivity.T, S=source.T)
psi = poisson.compute_psi()
psi[poisson.D==0]=np.nan
J = poisson.compute_flux_faces()
divJ = J["rp"] + J["rm"] + J["zp"] + J["zm"]

psi_mirror = np.flip(psi, axis=1)
psi = 0.5+psi/2
psi_mirror = 0.5-psi_mirror/2
psi = np.concatenate([psi, psi_mirror], axis=1)\

#J = {k:np.concatenate([v, np.flip(v, axis=1)], axis=1) for k,v in J.items()}

x,y=fields["xlayers"],fields["ylayers"]
x_,y_ = np.shape(psi)
crop_y = (y_-y)//2
crop_x = x_-x


# walls = fields["walls"]
# walls = np.pad(walls, ((crop_y,crop_y),(0,crop_x)), "edge")
fe = np.pad(fields["free_energy"].T, ((0,crop_x),(crop_y,crop_y)), "constant", constant_values = 0)

#psi[walls==True] = np.nan
c =  psi*np.exp(-fe)
#%%
gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 0.05], hspace=0.2, wspace=0, width_ratios=[1, 1])
fig = plt.figure()

arr2d = c

cmap_split = 0.5
cmap0_1 = cmr.get_sub_cmap("CMRmap_r", 0.0, cmap_split)
cmap1_max = cmr.get_sub_cmap("CMRmap_r", cmap_split, 1.0)


nr, nz = np.shape(psi)
extent = [-nz/2, nz/2, 0, nr]

ax = fig.add_subplot(gs[0, :])
c_arr_im = ax.imshow(
    arr2d,
    cmap=cmap0_1, 
    extent=extent, 
    origin = "lower",  
    aspect = 'equal',
    vmin = 0.0,
    vmax = 1.0,
    alpha = (arr2d<=1.0).astype(float),
    #norm = norm_,
    rasterized = True,
    )

gamma_=0.4
norm_ = mcolors.PowerNorm(
    gamma=gamma_, 
    vmin=1.0, 
    #vmax=np.nanmax(c_arr)
    vmax = np.nanmax(arr2d)
    ) 
c_arr_im2 = ax.imshow(
    arr2d,
    cmap=cmap1_max, 
    extent=extent, 
    origin = "lower",  
    aspect = 'equal',
    alpha = (arr2d>=1-1e-6).astype(float),
    norm = norm_,
    interpolation = "none",
    rasterized = True,
    )


width  = extent[1] - extent[0]
height = extent[3] - extent[2]

bg_rect = mpatches.Rectangle(
    (extent[0], extent[2]),  # bottom-left corner
    width,                    # width
    height,                   # height
    facecolor="green",        # fill color
    hatch="/",             # the hatching pattern
    edgecolor=None,
    zorder=-3                 # so it stays behind the imshow
)
ax.add_patch(bg_rect)

cax1 = fig.add_subplot(gs[1, 0])
cbar1 = fig.colorbar(c_arr_im, cax1, orientation = "horizontal")
cax2 = fig.add_subplot(gs[1, 1])
cbar2 = fig.colorbar(c_arr_im2, cax2, orientation = "horizontal")
cbar2.set_ticks([2, 5,10, 20, 30, 50, 70])
# cbar_im1 = plt.colorbar(c_arr_im2)

# #levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
# levels = [5e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3][::-1]
# #1e-2 gives an island
# levels = levels[:-1]+list(1-np.array(levels)[::-1])
# cs = ax.contour(arr2d, levels, 
#                 #colors = "black", 
#                 extent = extent,
#                 linewidth = 0.5
#                 )
# # Add labels to contours
# ax.clabel(cs, inline=False, fontsize=8)

#levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
#levels = [1e-2][::-1]
#1e-2 gives an island
#levels = levels[:-1]+list(1-np.array(levels)[::-1])
levels = [0.002, 0.005, 0.01, 0.02, 0.05, 0.5, 0.995, 0.998, 1]
# levels = [0.5, 1, 2, 5, 10, 15, 20]
cs = ax.contour(arr2d, levels, 
                colors = "green", 
                extent = extent,
                linewidth = 0.5
                )
# Add labels to contours
ax.clabel(cs, inline=False, fontsize=8)

levels = [2, 5, 10, 20, 30, 50]
cs = ax.contour(arr2d, levels, 
                colors = "red", 
                extent = extent,
                linewidth = 0.5
                )
# Add labels to contours
ax.clabel(cs, inline=False, fontsize=8)



ax.set_xlim(-150, 150)
ax.set_ylim(0,150)

wall_thickness = fields["s"]
pore_radius = fields["r"]

p = mpatches.Rectangle(
    (-wall_thickness/2, pore_radius), 
    wall_thickness, nr-pore_radius, 
    facecolor = "k", 
    edgecolor = "none", 
    #hatch ='/'
    alpha=0.2,
    )
ax.add_patch(p)

start_points_y = np.arange(0, pore_radius)[::2]
start_points_x = np.zeros_like(start_points_y)
start_points = np.array([start_points_x, start_points_y]).T

ax.streamplot(
    np.arange(-nz//2,0)+1, 
    np.arange(nr), 
    J["zm"], J["rm"],
    broken_streamlines=False,
    start_points = start_points,
    arrowsize = 0,
    color = "k",
    linewidth = 0.5
    )

fig.set_size_inches(6,4)
fig.savefig("fig/streamlines.svg", dpi = 600)
# %%
