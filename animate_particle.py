#%%
import cmasher as cmr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools

import numpy as np
import utils
import pandas as pd
from scipy.signal import convolve
import sfbox_utils
import seaborn as sns

matplotlib.rc('hatch', color='darkgreen', linewidth=9)

def mask_field(field, key ="phi", r_cut = None, z_cut = None, mirror = True):
    try:
        arr = field[key].squeeze()
    except KeyError:
        arr = field.dataset[key].squeeze()
    wall_thickness = field["s"].squeeze()
    pore_radius = field["r"].squeeze()
    l1 = field["l1"].squeeze()
    W_arr = np.zeros_like(arr)
    W_arr[pore_radius:, l1:l1+wall_thickness] = True
    z_center = l1+wall_thickness/2

    try:
        pc = field["pc"].squeeze()
        ph = field["ph"].squeeze()
        pw = field["pw"].squeeze()
        W_arr[:int(pw/2), int(z_center+pc-ph/2):int(z_center+pc+ph/2)] = True
    except KeyError:
        print("no particle")

    if r_cut is None:
        r_cut = np.shape(arr)[0]
    if z_cut is None:
        z_cut = np.shape(arr)[1]/2
    lz = int(z_center-z_cut)
    rz = int(z_center+z_cut)
    
    arr_masked = np.ma.array(arr, mask = W_arr)
    arr_masked = arr_masked[0:r_cut, lz:rz]
    
    if mirror:
        arr_masked = np.ma.vstack((np.flip(arr_masked), arr_masked[:,::-1]))[:,::-1]
        extent = [-z_cut, +z_cut, -r_cut, r_cut]
    else:
        extent = [-z_cut, +z_cut, 0, r_cut]
    return arr_masked, extent

def plot_frame_imshow(df, pc, key, 
            ax = None, cax = None,
            imshow_kwargs = {}, contour_kwargs = {},
            cmap = "viridis", bad_color = "green",
            r_cut = None, z_cut = None, mirror = True,
            contour = True, ):
    df_ = df.loc[df.pc == pc]
    if ax is None:
        fig, ax = plt.subplots()
    field, extent = mask_field(df_, key, r_cut, z_cut, mirror)
    if isinstance(cmap, str):
        cmap_ = matplotlib.colormaps[cmap]
    else:
        cmap_ = cmap
    cmap_.set_bad(color=bad_color)
    im = ax.imshow(
        field,
        origin = "lower",
        cmap = cmap_,
        extent = extent,
        interpolation = "none",
        **imshow_kwargs
    )

    if contour_kwargs is not None:
        contour = ax.contour(field,
            origin = "lower",
            extent = extent,
            **contour_kwargs
            )

    if cax is not None:
        cbar = plt.colorbar(im, cax = cax, 
            #orientation='horizontal',
            #shrink = 0.8,
            )
        if contour_kwargs is not None:
            cbar.add_lines(contour)
    else:
        cbar = None
        
    if contour_kwargs is not None:
        if (not "cmap" in contour_kwargs):
            contour_kwargs["cmap"] = cmap_.reversed()  
        if "colors" in contour_kwargs:
            del contour_kwargs["cmap"]
    
    s = df_.s.squeeze()
    pore_radius = df_.r.squeeze()
    pc = df_.pc.squeeze()
    ph = df_.ph.squeeze()
    pw = df_.pw.squeeze()
    p = mpatches.Rectangle((-s/2, -r_cut), s, r_cut-pore_radius, hatch='/', facecolor = "green")
    ax.add_patch(p)
    p = mpatches.Rectangle((-s/2, r_cut), s, -r_cut+pore_radius, hatch='/', facecolor = "green")
    ax.add_patch(p)

    #if abs(pc)<r_cut:
    p = mpatches.Rectangle((pc-ph/2, -pw/2), ph, pw, facecolor = "green", edgecolor = "black", linewidth = 0.7)
    ax.add_patch(p)
    ax.plot((pc-ph/2, pc+ph/2), (-pw/2, +pw/2), color = "black", linewidth = 0.7)
    ax.plot((pc-ph/2, pc+ph/2), (+pw/2, -pw/2), color = "black", linewidth = 0.7)

    ax.set_xlim(-z_cut, z_cut)
    ax.set_ylim(-r_cut, r_cut)

    yticks_labels=[]
    for label in ax.get_yticklabels():
        text = label.get_text()
        text = text[1:] if text.startswith('−') else text
        yticks_labels.append((text))
    ax.set_yticklabels(yticks_labels)

    return ax, cbar

def add_to_profile(x_arr, y_arr, x, y):
    itemindex = np.argmax(x_arr>x)
    x_arr_ = np.insert(x_arr, itemindex, x)
    y_arr_ = np.insert(y_arr, itemindex, y)
    return x_arr_, y_arr_
#%%
s = 52
r = 26
d=8
ph =d
pw =d
sigma = 0.02
chi_PS = 0.5
chi_PC = -0.5
X = None

master = pd.read_pickle("reference_table.pkl")
#master = master.loc[master["comment"] == "grown_from_small"]
master_empty = pd.read_pickle("reference_table_empty_brush.pkl")
master_empty = master_empty.loc[
    (master_empty.s == s) \
    & (master_empty.r== r) \
    & (master_empty.sigma == sigma) #\
    & (master_empty.chi_PS == chi_PS)
    ]
master = master.loc[master.chi_PC==chi_PC]
master = master.loc[master.ph==ph]
master = master.loc[master.chi_PS==chi_PS]
master = master.loc[master.sigma == sigma]
master.sort_values(by = "pc", inplace = True)
#%%
master["phi"] = master.dataset["phi"]
phi0 = master_empty.dataset["phi"].squeeze()
master["delta_phi"] = master.apply(lambda _: _["phi"] - phi0, axis = 1)
# %%
r_cut = 60
z_cut = 60

fig  = plt.figure(
        figsize = (8,4.3),
        dpi=600
        )

gs = GridSpec(
    3, 5, 
    figure=fig, 
    height_ratios=[
        0.4,
        1.0,   # phi & delta_phi
        0.2,  # spacer
        # 0.05,  # spacer
        # 0.05,  # delta_phi_x part 1
        # 0.05,  # delta_phi_x part 2
        # 0.05   # delta_phi_x part 3
    ], 
    width_ratios=[
        1,  # col 0: phi & colorbars
        0.4,
        0.1,
        1,
        0.4   # col 4: delta_phi_y
    ],
    hspace=0.1,
    wspace=0.1
)

pc_list = np.array(master.pc.squeeze())
pc_list = pc_list[abs(pc_list)<z_cut]
def update_figure(i):
    fig.clear()
    pc = pc_list[i]

    ax_0 = fig.add_subplot(gs[1, 0])  # phi
    ax_1 = fig.add_subplot(gs[1, 3], sharex=ax_0, sharey=ax_0)


    pcax0 = mpatches.Rectangle(
        (-60, -60), 120, 60, 
        facecolor = "white", 
        edgecolor = "none",
        alpha = 0.95,
        )
    # Inset colorbars
    cax0 = inset_axes(
        ax_0,
        width="5%",  # width = 5% of parent_bbox width
        height="80%",  # height = 50% of parent_bbox height
        loc='lower left',
        bbox_to_anchor=(0.95, 0.1, 1, 1),
        bbox_transform=ax_0.transAxes,
        borderpad=0,
    )

    cax1 = inset_axes(
        ax_1,
        width="5%",  # width = 5% of parent_bbox width
        height="80%",  # height = 50% of parent_bbox height
        loc='lower left',
        bbox_to_anchor=(0.95, 0.1, 1, 1),
        bbox_transform=ax_1.transAxes,
        borderpad=0,
    )

    cax1.set_xticks([-0.01, 0, 0.01])
    #cax1.set_yticklabels(["Min", "Mid", "Max"])
    
    axd = {
        "phi": ax_0,
        #"cax0": fig.add_subplot(gs[0, 1]),
        "cax0": cax0,
        #"cax1": fig.add_subplot(gs[2, 1]),
        "cax1": cax1,
        "delta_phi": ax_1,
        "delta_phi_x": fig.add_subplot(gs[0, 3], sharex=ax_0),
        "delta_phi_y": fig.add_subplot(gs[1, 4], sharey=ax_0),
        "phi_x": fig.add_subplot(gs[0, 0], sharex=ax_0),
        "phi_y": fig.add_subplot(gs[1, 1], sharey=ax_0),
    }



    phi = master.loc[master.pc == pc]["phi"].squeeze()
    fe = master.loc[master.pc == pc]["free_energy"].squeeze()
    delta_phi = phi - phi0
    wall_thickness = master_empty["s"].squeeze()
    pore_radius = master_empty["r"].squeeze()
    l1 = master_empty["l1"].squeeze()
    z_center = l1+wall_thickness/2



    axd["phi"].set_xlabel("z")
    axd["delta_phi"].set_xlabel("z")
    axd["phi"].set_ylabel("r")
    _, cbar = plot_frame_imshow(master, pc, 
                    ax = axd["phi"],
                    cax = axd["cax0"],
                    key = "phi", 
                    imshow_kwargs=dict(vmin = 0, vmax = 0.35),
                    cmap = "gnuplot2_r", 
                    contour_kwargs =dict(
                        linewidths = 0.2, 
                        levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
                        colors = "black"
                    ),
                    bad_color ="darkgreen", 
                    r_cut=r_cut, z_cut = z_cut
                    )
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.yaxis.set_ticks_position('left')
    # cbar.ax.set_ylabel(
    #     cax_label, rotation = "horizontal", #labelpad=15,
    #     ha = "left", va = "center",
    #     )

    #axd["delta_phi"].set_xlabel("z")
    #axd["delta_phi"].set_xticklabels([])

    cmap0 = cmr.get_sub_cmap("seismic", 0.0, 0.5)
    cmap1 = cmr.get_sub_cmap("seismic", 0.5, 1.0)
    # if chi_PC<= -1.0: vmin, vmax = -0.025, 0.1
    # else: vmin, vmax = -0.1, 0.025
    vmin, vmax = -0.01, 0.01
    levels = np.arange(vmin, vmax+0.025, 0.025)
    cmap_ = cmr.combine_cmaps(cmap0, cmap1, nodes=[(0-vmin)/(vmax-vmin)])

    _, cbar = plot_frame_imshow(master, pc, 
                    ax = axd["delta_phi"],
                    cax = axd["cax1"],
                    #cax_label="$\Delta\phi$",
                    key = "delta_phi", 
                    imshow_kwargs=dict(vmin = vmin, vmax = vmax),
                    cmap = cmap_, 
                    contour_kwargs =dict(
                        linewidths = 0.2,
                        levels = levels,
                        colors = "black"
                    ),
                    #contour_kwargs = None,
                    bad_color ="darkgreen", 
                    r_cut=r_cut, z_cut = z_cut
                    )
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-0.01, 0, 0.01])

    axd["phi_x"].yaxis.tick_right()
    axd["phi_x"].set_ylabel("$\phi$", rotation = "horizontal", labelpad = 5)
    axd["phi_x"].yaxis.set_label_position("right")
    #axd["delta_phi_x"].set_ylim(0.35, 0)
    #axd["delta_phi_x"].set_ylim(0, 0.35)
    axd["phi_x"].set_ylim(0, 0.35)
    axd["phi_x"].set_yticks([0.1, 0.2, 0.3])


    axd["delta_phi_x"].yaxis.tick_right()
    axd["delta_phi_x"].set_ylabel("$\Delta\phi$", rotation = "horizontal", labelpad = 5)
    axd["delta_phi_x"].yaxis.set_label_position("right")
    axd["delta_phi_x"].set_ylim(-0.15, 0.15)

    x_arr = np.arange(int(-z_cut), int(+z_cut))+0.5
    delta_phi_x=phi[0, int(z_center-z_cut):int(z_center+z_cut)]
    phi_0_x = phi0[0, int(z_center-z_cut):int(z_center+z_cut)]

    x_arr_, delta_phi_x_ = add_to_profile(x_arr, delta_phi_x, pc-ph/2-0.5, 0)
    x_arr_, delta_phi_x_ = add_to_profile(x_arr_, delta_phi_x_, pc+ph/2, 0)

    x_arr_, phi_0_x_ = add_to_profile(x_arr, phi_0_x, pc-ph/2-0.5, 0)
    x_arr_, phi_0_x_ = add_to_profile(x_arr_, phi_0_x_, pc+ph/2, 0)

    axd["phi_x"].plot(x_arr_, delta_phi_x_)
    axd["phi_x"].plot(x_arr, phi_0_x, 
                            color = "black", linewidth = 0.5,
                            )
        
    axd["delta_phi_x"].plot(x_arr, delta_phi_x - phi_0_x)
    # axd["delta_phi_x"].plot(x_arr, phi_0_x, 
    #                         color = "black", linewidth = 0.5,
    #                         )


    delta_phi_y = phi[:r_cut, int(z_center+pc)][::-1]
    delta_phi_y = np.concatenate([delta_phi_y, delta_phi_y[::-1]])

    phi_0_y = phi0[:r_cut, int(z_center+pc)][::-1]
    phi_0_y = np.concatenate([phi_0_y, phi_0_y[::-1]])

    y_arr = np.arange(int(-r_cut), int(+r_cut), 1) + 0.5
    y_arr_, delta_phi_y_ = add_to_profile(y_arr, delta_phi_y, -pw/2-0.5, 0)
    y_arr_, delta_phi_y_ = add_to_profile(y_arr_, delta_phi_y_,  pw/2, 0)

    y_arr_, phi_0_y_ = add_to_profile(y_arr, phi_0_y, -pw/2-0.5, 0)
    y_arr_, phi_0_y_ = add_to_profile(y_arr_, phi_0_y_,  pw/2, 0)
    #x_arr_, delta_phi_x_ = add_to_profile(x_arr_, delta_phi_x_, pc+ph/2, 0)

    axd["phi_y"].plot(delta_phi_y_,y_arr_)
    axd["phi_y"].plot(phi_0_y,y_arr, 
                        color = "black", linewidth = 0.5)
    
    axd["delta_phi_y"].plot(delta_phi_y-phi_0_y,y_arr)


    axd["delta_phi_y"].set_xlim(-0.15, 0.15)
    #axd["delta_phi_y"].set_xticks([0, 0.1, 0.2, 0.3])
    axd["delta_phi_y"].xaxis.set_tick_params(rotation=-90)
    axd["delta_phi_y"].set_xlabel("$\Delta\phi$",labelpad = 0)

    axd["phi_y"].set_xlim(0, 0.35)
    axd["phi_y"].set_xticks([0, 0.1, 0.2, 0.3])
    axd["phi_y"].xaxis.set_tick_params(rotation=-90)
    axd["phi_y"].set_xlabel("$\phi$",labelpad = 0)


    #cross
    axd["delta_phi"].axvline(pc, linewidth = 0.3, linestyle = "--", color = "black")
    axd["delta_phi_x"].axvline(pc, linewidth = 0.3, linestyle = "--", color = "black")
    axd["delta_phi"].axhline(0, linewidth = 0.3, linestyle = "--", color = "black")
    axd["delta_phi_y"].axhline(0, linewidth = 0.3, linestyle = "--", color = "black")

    axd["delta_phi_x"].axhline(0, linewidth = 0.3, linestyle = "-", color = "black")
    axd["delta_phi_y"].axvline(0, linewidth = 0.3, linestyle = "-", color = "black")

    #axd["delta_phi"].axvline(pc, linewidth = 0.3, linestyle = "--", color = "black")
    axd["phi_x"].axvline(pc, linewidth = 0.3, linestyle = "--", color = "black")
    #axd["phi"].axhline(0, linewidth = 0.3, linestyle = "--", color = "black")
    axd["phi_y"].axhline(0, linewidth = 0.3, linestyle = "--", color = "black")

    plt.setp(axd["delta_phi"].get_yticklabels(), visible=False)
    plt.setp(axd["delta_phi_y"].get_yticklabels(), visible=False)
    plt.setp(axd["phi_y"].get_yticklabels(), visible=False)
    #plt.setp(axd["delta_phi"].get_xticklabels(), visible=False)
    plt.setp(axd["delta_phi_x"].get_xticklabels(), visible=False)
    plt.setp(axd["phi_x"].get_xticklabels(), visible=False)

    print(i)

    fe_str = fr"$\Delta F / k_B T = {fe:.2f}$" if fe < 0 else f"$\Delta F / k_B T = +{fe:.2f}$"
    axd["delta_phi"].text(-55, -54, s = fe_str, ha = "left", 
                    #bbox=dict(facecolor='white', edgecolor='black')
                    )
    pc_str = fr"$z_c = {pc}$" if pc < 0 else f"$z_c = +{pc}$"
    axd["phi"].text(-55, -54, s = pc_str, ha = "left", 
                #bbox=dict(facecolor='white', edgecolor='black')
                )
    
    p = mpatches.Rectangle((-58, -59), 71, 15, facecolor = "white", edgecolor = "k")
    axd["delta_phi"].add_patch(p)

    p = mpatches.Rectangle((-58, -59), 44, 15, facecolor = "white", edgecolor = "k")
    axd["phi"].add_patch(p)

    axd["phi"].text(-56, 48, s = "$\phi$", ha = "left", 
                    fontsize = 14,
                    #bbox=dict(facecolor='white', edgecolor='black')
                    )
    axd["delta_phi"].text(-56, 48, s = "$\Delta \phi$", ha = "left", 
                    fontsize = 14,
                    #bbox=dict(facecolor='white', edgecolor='black')
                    )
    
    axd["delta_phi_y"].text(0.1, 0.96, s = "$\Delta\phi_{z=z_c}$", ha = "left",va="top", 
                fontsize = 14,
                transform = axd["delta_phi_y"].transAxes,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor='white', edgecolor='none', 
                          alpha= 0.9
                          )
                )
    
    axd["delta_phi_x"].text(0.95, 0.9, s = "$\Delta\phi_{r=0}$", 
            ha = "right",va="top", 
            fontsize = 14,
            transform = axd["delta_phi_x"].transAxes,
            bbox=dict(boxstyle="round,pad=0.2",
                        facecolor='white', edgecolor='none', 
                        alpha= 0.9)
            )
    
    axd["phi_y"].text(0.1, 0.96, s = "$\phi_{z=z_c}$", ha = "left",va="top", 
                fontsize = 14,
                transform = axd["phi_y"].transAxes,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor='white', edgecolor='none', 
                          alpha= 0.9
                          )
                )
    
    axd["phi_x"].text(0.95, 0.9, s = "$\phi_{r=0}$", 
            ha = "right",va="top", 
            fontsize = 14,
            transform = axd["phi_x"].transAxes,
            bbox=dict(boxstyle="round,pad=0.2",
                        facecolor='white', edgecolor='none', 
                        alpha= 0.9)
            )
    
    fig.text(
        0.1, 0.05,
f"""SUPPLEMENTARY MOVIE. Scheutjens-Fleer SCF. Cylindrical particle insertion and perturbation to the polymer density.
Colloid diameter and height: {pw}; pore radius: {pore_radius}; pore length: {wall_thickness}; segment length: 1; number of segments: {300}; grafting density: {sigma};
polymer-solvent interaction parameter: {chi_PS};  polymer-coloid interaction parameter: {chi_PC};""",
fontsize = 6,
fontdict={"family":"monospace"}
    )


    #ax_0.add_patch(pcax0)

update_figure(2)
fig
 # %%
import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, update_figure, frames=len(pc_list))
FFwriter = animation.FFMpegWriter(fps = 10)
# %%
mp4_name = f"media/insertion_{chi_PS=}_{chi_PC=}_{d=}.mp4"
ani.save(mp4_name, writer=FFwriter, dpi =600)
# %%
i=2
update_figure(i)
fig.savefig(f"fig/insertion_{chi_PS=}_{chi_PC=}_{d=}_pc={pc_list[i]}.svg")
# %%
