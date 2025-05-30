#%%
import numpy as np
import matplotlib.pyplot as plt
import calculate_fields_in_pore
import os
from collections import defaultdict
here = os.path.dirname(__file__)

import matplotlib.style as style
from matplotlib import rc, rcParams
import matplotlib.patches as mpatches
import matplotlib

rc('hatch', color='darkgreen', linewidth=9)

rcParams.update({
    "mathtext.fontset": "cm",  # Use Computer Modern
    "font.family": "serif",
})
#%%
def pad_fields(fields, pad_sides, pad_top):
    fields["xlayers"]=fields["xlayers"]+pad_sides*2
    fields["ylayers"]=fields["ylayers"]+pad_top

    fields["h"]=fields["h"]+pad_top
    fields["l1"]=fields["l1"]+pad_sides
    fields["l2"]=fields["l2"]+pad_sides

    # mode = defaultdict(lambda: {mode = "edge"})
    # mode["free_energy"] = 
    padding = ((pad_sides, pad_sides),(0, pad_top))

    for k in fields.keys():
        if k in ["walls", "mobility", "conductivity"]: continue
        try:
            fields[k] = np.pad(
                fields[k],
                padding, 
                "constant", constant_values=(0.0, 0.0)
                )
            print(k, "padded")
        except ValueError:
            pass
        
    fields["walls"]=np.pad(
        fields["walls"],
        padding,
        "edge",
        )
    print("walls", "padded")
    
    fields["mobility"]=np.pad(
        fields["mobility"],
        padding, 
        "constant", constant_values=(1.0, 1.0)
        )
    fields["mobility"][fields["walls"]==True]=0.0
    print("mobility", "padded")

    bulk = fields["conductivity"][1,1]
    fields["conductivity"]=np.pad(
        fields["conductivity"],
        padding, 
        "constant", constant_values=(bulk, bulk)
    )
    fields["conductivity"][fields["walls"]==True]=0.0
    print("conductivity", "padded")

def plot_walls(ax,
               fields,
               r_cut=None,
               z_cut=None,
               mirror=False
               ):
    wall_thickness = fields["s"]
    l1 = fields["l1"]
    xlayers = fields["xlayers"]
    ylayers = fields["ylayers"]
    pore_radius = fields["r"]
    alpha = 0.1
    if r_cut is None: r_cut = xlayers+1
    if z_cut is None: z_cut = ylayers+1
    if mirror:
        p = mpatches.Rectangle(
            (-wall_thickness/2, -r_cut), 
            wall_thickness, r_cut-pore_radius, 
            facecolor = "k", 
            edgecolor = "none", 
            #hatch ='/'
            alpha=alpha,
            )
        ax.add_patch(p)
    p = mpatches.Rectangle(
        (-wall_thickness/2, r_cut), 
        wall_thickness, -r_cut+pore_radius, 
        facecolor = "k", 
        edgecolor = "none", 
        #hatch ='/'
        alpha=alpha,
        )
    ax.add_patch(p)
    
def plot_heatmap(ax,
                 fields, 
                 key, 
                 r_cut=None, 
                 z_cut=None, 
                 vmin=None, 
                 vmax=None, 
                 cmap="viridis",
                 mirror=False
                 ):
    from heatmap_explorer import plot_heatmap_and_profiles
    wall_thickness = fields["s"]
    l1 = fields["l1"]
    xlayers = fields["xlayers"]
    ylayers = fields["ylayers"]
    pore_radius = fields["r"]

    if r_cut is None: r_cut = xlayers
    if z_cut is None: z_cut = int(ylayers/2)

    def cut_and_mirror(arr):
        cut = arr.T[0:r_cut, int(ylayers/2)-z_cut:int(ylayers/2)+z_cut]
        return np.vstack((np.flip(cut), cut[:,::-1])).T[::-1]
    if mirror:
        extent = [-z_cut, z_cut, -r_cut, r_cut]
        mask = cut_and_mirror(fields["walls"])
        array = cut_and_mirror(fields[key])
    else:
        extent = [-z_cut, z_cut, 0, r_cut]
        mask= fields["walls"][int(ylayers/2)-z_cut:int(ylayers/2)+z_cut, 0:r_cut] 
        array=fields[key][int(ylayers/2)-z_cut:int(ylayers/2)+z_cut, 0:r_cut] 
    array = np.ma.array(array, mask = mask)

    if vmin is None: vmin= np.nanmin(array)
    if vmax is None: vmax=np.nanmax(array)

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

    return im

#%%
a0, a1 = [0.7, -0.3]
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness
chi_PC = -1.3
chi_PS = 0.5
sigma = 0.02
d=12

fields = calculate_fields_in_pore.calculate_fields(a0, a1, chi_PC, chi_PS, wall_thickness, pore_radius, d, sigma)

xlayers= fields["xlayers"]
ylayers = fields["ylayers"]

# %%
fig, ax = plt.subplots()
# z_cut=100
# r_cut=60

bg = mpatches.Rectangle(
    (0, 0), 1, 1,               # (x, y), width, height in axes coordinates
    transform=ax.transAxes,    # makes it relative to axes (0-1 range)
    facecolor='green',          # transparent fill
    edgecolor='darkgreen',         # hatch color
    hatch='/',               # hatch pattern
    zorder=-10                 # draw below everything else
)
ax.add_patch(bg)

#cmap0_1 = cmr.get_sub_cmap("CMRmap_r", 0.0, 0.5)
#cmap1_max = cmr.get_sub_cmap("CMRmap_r", 0.5, 1.0)

plot_heatmap(ax, fields, "c", 
             #r_cut=r_cut, z_cut=z_cut, 
             cmap = "cividis", 
             mirror=False,
             vmin=0, vmax = 2)
plot_walls(ax, fields, 
           #r_cut, z_cut, 
           mirror=False)
ax.set_aspect("equal")


# J_arr = np.load(sim_name.replace(".txt", "_J_arr.npy"))

# x = np.arange(0, ylayers, 10)-int(ylayers/2)
# y = np.arange(0, xlayers, 10)
# xx, yy = np.meshgrid(x, y)
# uv = [J_arr[xx_+int(ylayers/2), yy_] for xx_, yy_ in zip(xx, yy)]
# u = np.moveaxis(uv, -1, 0)[0]
# v = np.moveaxis(uv, -1, 0)[1]
# norm = np.linalg.norm(np.array((u, v)), axis=0)


# # start_points_y = np.arange(0, pore_radius-1,1)
# # start_points_x = np.ones_like(start_points_y)#*dd_obj.zlayers/2
# start_points_y = np.arange(0, xlayers,20)
# start_points_x = np.zeros_like(start_points_y)-ylayers/2+1
# start_points = np.array([start_points_x, start_points_y]).T
# J_arr_stream = ax.streamplot(
#     xx, yy, u, v, 
#     #color = norm,
#     color = "k",
#     start_points = start_points,
#     broken_streamlines = False,
#     arrowsize = 0,
#     linewidth = 0.3,
#     integration_direction="forward"
#     )

levels = [0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ax.contour(fields["c"].T,levels=levels, extent = [-ylayers/2, ylayers/2, 0, xlayers], colors = "red")

levels = np.arange(0.00, 0.1, 0.02)
ax.contour(fields["c"].T,levels=levels, extent = [-ylayers/2, ylayers/2, 0, xlayers], colors = "orange")
levels = np.arange(0.90, 1.0, 0.02)
ax.contour(fields["c"].T,levels=levels, extent = [-ylayers/2, ylayers/2, 0, xlayers], colors = "orange")

#ax.set_xlim(-100,100)
#ax.set_ylim(0,100)

# %%
