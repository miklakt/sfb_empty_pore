#%%
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from utils import get_by_kwargs, load_datasets
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import theory
import scf_pb
LAST_USED_COLOR = lambda: plt.gca().lines[-1].get_color()
#%%

def particle_viewport(array2d, pc, viewport, ph =None, pw = None):
    x0 = int(pc-viewport[1])
    x1 = int(pc+viewport[1])
    y0 = int(0)
    y1 = int(viewport[0])
    view = array2d[y0:y1, x0:x1]
    return view

def phi_change_vicinity(phi, phi_empty, pc, viewport = (20, 10), ph =None, pw = None):
    delta_phi = particle_viewport(phi, pc, viewport)-particle_viewport(phi_empty, pc, viewport, ph, pw)
    if ph is not None:
        delta_phi[int(0):int(pw/2), int(viewport[1]-ph/2):int(viewport[1]+ph/2)] = np.nan
    return delta_phi

def imshow_data(fig, ax, data, contour = True, imshow_kwargs = {}, mirror = False, colorbar = True, cbar_title = None):
    imshow_kwargs_default = dict(origin = "lower", aspect = "equal", cmap = "RdBu_r")
    imshow_kwargs_default.update(imshow_kwargs)

    cmap = plt.cm.get_cmap(imshow_kwargs_default["cmap"]).copy()
    cmap.set_bad('darkgreen')
    imshow_kwargs_default["cmap"] = cmap

    if mirror:
        data = np.vstack([np.flip(data, axis = 0), data])
    
    c = ax.imshow(data, **imshow_kwargs_default)

    if contour:
        ax.contour(
            data,
            colors = 'black',
            alpha = 1,
            linestyles = 'solid',
            linewidths = 0.4,
            levels = np.arange(0, 0.30, 0.04)
        )
    ax.set_aspect(aspect = 1)
    if colorbar:
        cax = ax.inset_axes([0.95, 0.0, 0.05, 1.0], transform=ax.transAxes)
        cbar = fig.colorbar(c, ax=ax, cax=cax)
        if cbar_title is not None:
            cbar.set_label(cbar_title)

    return c


def draw_particle(fig, ax, pc, ph, pw, xlayers = None, mirror = False, **patch_kwargs):
    if mirror:
        x0 = pc-ph/2
        y0 = xlayers-pw/2
        dx = ph-1
        dy = pw-1
    else:
        x0 = pc-ph/2
        y0 = 0
        dx = ph-1
        dy = pw/2-1

    rect = patches.Rectangle((x0, y0), dx, dy, **patch_kwargs)
    ax.add_patch(rect)

def draw_pore_walls(fig, ax, l1, h, s, xlayers, mirror = False, **patch_kwargs):
    if mirror:
        x0 = l1
        y0 = 0
        dx = s-1
        dy = h-1
        rect = patches.Rectangle((x0, y0), dx, dy, **patch_kwargs)
        ax.add_patch(rect)

        x0 = l1
        y0 = xlayers*2-h
        dx = s-1
        dy = h-1
        rect = patches.Rectangle((x0, y0), dx, dy, **patch_kwargs)
        ax.add_patch(rect)

    else:
        x0 = l1
        y0 = xlayers-h
        dx = s-1
        dy = h-1
        rect = patches.Rectangle((x0, y0), dx, dy, **patch_kwargs)
        ax.add_patch(rect)

def draw_hm_frame(fig, ax, datum, mirror = False, imshow_kwargs = {}, patches_kwargs = {}, colorbar = True):
    imshow_data(
        fig, ax, datum.phi, contour=True, 
        mirror=mirror,
        imshow_kwargs=imshow_kwargs,
        colorbar=colorbar,
        cbar_title='$\phi$'
        )
    draw_particle(
        fig, ax, datum.pc, datum.ph, datum.pw, xlayers=datum.xlayers, 
        mirror=mirror,
        **patches_kwargs
        )
    draw_pore_walls(
        fig, ax, datum.l1, datum.h, datum.s, datum.xlayers,
        mirror=mirror,
        **patches_kwargs
        )

def draw_free_energy_frame(fig, ax, z, fe, pc_sfb, fe_sfb , **kwargs):
    ax.plot(z, fe, label ="model", zorder = -1,)
    ax.scatter(
        pc_sfb, fe_sfb, 
        color = "green",
        label ="numeric",
        **kwargs,
        )
    ax.legend(loc = "upper left")

def draw_current_free_energy(fig, ax, z, fe, **kwargs):
    ax.scatter(
        z, fe, 
        color = "green",
        **kwargs,
    )

def draw_layout(fig, axs, title = None):
    axs[0].set_ylabel("$r$")
    axs[0].set_xlabel("$z$")
    axs[1].set_ylabel("$F/k_BT$")
    axs[1].set_xlabel("$z$")
    if title is not None:
        fig.suptitle(title)

def draw_phi_vicinity(fig, axins, delta_phi, ph, pw, vmin, vmax):

    imshow_data(
        fig, axins, delta_phi, contour=False, 
        mirror=False,
        imshow_kwargs=dict(vmin = vmin, vmax=vmax, cmap = "cividis"),
        #colorbar=colorbar,
        #cbar_title='$\Delta \phi$',
    )
    shape = delta_phi.shape
    rect = patches.Rectangle((shape[0]-ph/2-0.5, -0.5), ph, pw/2, facecolor = "green", edgecolor = "red")
    axins.add_patch(rect)

def animate_factory(
        fig, axs, 
        data, 
        z, fe,
        title = None,
        **kwargs
        ):
    init = True
    delta_phi_min = np.min(data.delta_phi.apply(np.nanmin))
    delta_phi_max = np.max(data.delta_phi.apply(np.nanmax))
    #delta_phi_min = -0.005
    #delta_phi_max = +0.005
    def animate(i):
        nonlocal init
        [ax.cla() for ax in axs]
        datum = data.iloc[i]
        draw_hm_frame(
            fig, axs[0], 
            datum, 
            #colorbar = init, 
            **kwargs
            )
        draw_free_energy_frame(
            fig, 
            axs[1], 
            z, 
            fe, 
            data.pc, 
            data.free_energy,
            s = 3, 
            marker = "s", 
            facecolors='green', 
            edgecolors='green'
            )
        draw_current_free_energy(
            fig, 
            axs[1], 
            datum.pc, 
            datum.free_energy,
            s = 10, 
            marker = "s", 
            facecolors='green',
            edgecolors='red'
            )
        axins = inset_axes(axs[1], width="50%", height="50%", loc=5, borderpad =-3)
        axins.set_xticks([], major = False)
        axins.set_yticks([], major = False)
        axins.margins(0)
        axins.set_title("$\Delta \phi$")
        draw_phi_vicinity(
            fig, 
            axins, 
            datum.delta_phi,
            datum.ph, 
            datum.pw,
            vmin = delta_phi_min,
            vmax = delta_phi_max
            )
        draw_layout(fig, axs, title)
        init = False
    return animate
#%%
#chi_ps = []
chi_pc = -1
s = 52 
r = 26
a1, a2 = [ 0.19814812, -0.08488959]
ph = 4
pw = 4
particle_in_pore = pd.read_pickle("reference_table.pkl")
empty_pore = pd.read_pickle("empty_brush.pkl")
selected_data = get_by_kwargs(
    particle_in_pore,
    chi_PC =chi_pc,
    s=s,
    r=r,
    ph=ph,
    pw = pw
    )
empty_pore = get_by_kwargs(empty_pore, s = s, r = r).squeeze()


fig, ax = plt.subplots()

selected_data.sort_values(by = ["pc", "chi_PS"], inplace = True)
for idx, group in selected_data.groupby(by = "chi_PS"):
    print(idx)
    phi_center = get_by_kwargs(empty_pore, chi_PS = idx, s = s, r = r).squeeze().phi[0]
    phi_center = np.array_split(phi_center,2)[0]
    fe = np.array([theory.free_energy_phi(
                phi = phi_,
                a1=a1, a2=a2, 
                chi_PC=chi_pc, 
                chi_PS=idx, 
                w=pw
            ) for phi_ in phi_center])
    z = range(len(fe))
    ax.plot(z, fe, label = idx)
    ax.plot(group.pc, group.free_energy, color = LAST_USED_COLOR(), linewidth = 0, marker = "o", markerfacecolor = "none")
    
legend2 = plt.legend([Line2D([0],[0], color = 'k'), Line2D([0],[0], color = 'k', linewidth=0,  marker = "o", markerfacecolor = "none")], ['SS-SCF', "SF-SCF"])
ax.add_artist(legend2)

ax.legend(title = "$\chi_{PS}$")

ax.set_ylabel("$F/k_BT$")
ax.set_xlabel("$z$")

parameters_title = "$\chi_{PC} = $" + f"{chi_pc}, " +\
    "$h = $" + f"{ph}, " +\
    "$r_{pore}$ = " + f"{r}"
ax.set_title("Nanocolloid in a pore. Insertion free energy\n" + parameters_title)
# %%
fig.savefig("conference/insertion_free_energy_pore.png")
# %%
fig, axs = plt.subplots(
    nrows = int(len(empty_pore.chi_PS.unique())/2), 
    ncols = 2,
    sharex=True, 
    sharey=True
    )
plt.subplots_adjust(hspace=0.1, wspace=0.1, right=0.8)
fig.set_size_inches(7, 10)
for ax, (idx, group) in zip(axs.flat, empty_pore.groupby(by = "chi_PS")):
    group = group.squeeze()
    view = slice(int(group.ylayers/2-60), int(group.ylayers/2+60))
    im = imshow_data(fig, ax, group.phi[...,view], imshow_kwargs=dict(vmin=0, vmax=0.4), colorbar=False)
    draw_pore_walls(fig, ax, group.l1-86, group.h, group.s, group.xlayers, color = "green")
    
    ax.text(0.5, 0.98, "$\chi_{PS} = $"+f"{idx}", transform = ax.transAxes, ha="center", va = "top", color = "white")

cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

fig.savefig("conference/phi_heatmaps.pdf")
# %%
