# %%
import matplotlib.animation as animation
import utils
import sfbox_utils
from theory import free_energy_phi
from sfbox_utils.utils import ld_to_dl
import pandas as pd
import numpy as np
import ascf_pb
from utils import get_by_kwargs, ground_energy_correction, load_datasets
from plot_utils import plot_imshow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import theory
from scipy.interpolate import interp1d

chi_ps = 0.5
chi_pc = -1.0
s = 52
r = 26
fig, axs = plt.subplots(
    nrows=2, sharex=True,
    gridspec_kw=dict(width_ratios=(1,), height_ratios=(1.6, 1))
)
a1, a2 = [0.16814812, 0.08488959]
d = 4
ph = d
pw = d


df_empty = pd.read_pickle("empty_brush.pkl")
df_empty = get_by_kwargs(df_empty, chi_PS=chi_ps, s=s, r=r)
phi_empty = df_empty["phi"].squeeze()
phi_center = phi_empty[0]
# %%
kernel = theory.gauss_kernel_cyl(m=d*4, sigma=d/2)
# %%
phi_corrected = theory.correct_phi(phi_empty, kernel)

plt.plot(phi_center, label="r=0")
plt.plot(phi_corrected, label="corrected")
plt.legend()

# %%
fe_corrected = np.array([free_energy_phi(
    phi=phi_,
    a1=a1, a2=a2,
    chi_PC=chi_pc,
    chi_PS=chi_ps,
    w=d
    # ) for phi_ in phi_center])
) for phi_ in phi_corrected])

fe = np.array([free_energy_phi(
    phi=phi_,
    a1=a1, a2=a2,
    chi_PC=chi_pc,
    chi_PS=chi_ps,
    w=d
    # ) for phi_ in phi_center])
) for phi_ in phi_center])
phi_cb = interp1d(range(len(phi_center)), phi_center)
def Pi_cb(_): return theory.Pi(phi_cb(_), chi_ps)


surface_integrand = ascf_pb.particle_geometry.cylinder.surface_integrand(
    ph=ph, pw=pw)

volume_integrand = ascf_pb.particle_geometry.cylinder.volume_integrand(
    ph=ph, pw=pw)

fe_ascf_func = np.vectorize(lambda _: ascf_pb.free_energy.total_free_energy(
    _-ph/2, _+ph/2, surface_integrand, volume_integrand, phi_cb, Pi_cb, chi_ps, chi_pc, [a1, a2])*2)

z_ascf = np.arange(ph, (len(phi_center)-ph)/2)
fe_ascf = fe_ascf_func(z_ascf)


# %%
fe = np.split(fe, 2)[0]
fe_corrected = np.split(fe_corrected, 2)[0]


def plot_imshow(dataframe, array_name, contour=True, imshow_kwargs={}, patch=True, patches_kwargs={}, colorbar=True,  **kwargs):
    imshow_kwargs_default = dict(origin="lower", aspect="equal", cmap="RdBu_r")
    imshow_kwargs_default.update(imshow_kwargs)

    print(kwargs)

    plot_data = get_by_kwargs(dataframe, **kwargs).squeeze()
    X = plot_data[array_name].reshape(
        (plot_data["xlayers"], plot_data["ylayers"]))
    # X_m = np.vstack([np.flip(X, axis = 0),X])
    X_m = np.hsplit(X, 2)[0]
    # X_m = X

    c = axs[0].imshow(X_m, **imshow_kwargs_default)
    if contour:
        axs[0].contour(
            X_m,
            colors='black',
            alpha=1,
            linestyles='solid',
            linewidths=0.4,
            levels=np.arange(0, 0.30, 0.04)
        )
    axs[0].set_aspect(aspect=1)
    axs[0].set_title(" ".join([str(k)+"="+str(v) for k, v in kwargs.items()]))

    if colorbar:
        plt.colorbar(
            c, ax=axs,
            orientation="horizontal",
            location="top",
            shrink=0.6
        )

    if patch:
        # rect = patches.Rectangle((plot_data.l1-1, 0), plot_data.s, plot_data.h-1, **patches_kwargs)
        # axs[0].add_patch(rect)

        # rect = patches.Rectangle((plot_data.l1-1, plot_data.xlayers*2-plot_data.h), plot_data.s, plot_data.h-1, **patches_kwargs)
        rect = patches.Rectangle((plot_data.l1, plot_data.xlayers-plot_data.h),
                                 plot_data.s/2-1, plot_data.h-1, **patches_kwargs)
        # rect = patches.Rectangle((plot_data.l1, plot_data.xlayers-plot_data.h), plot_data.s-1, plot_data.h-1, **patches_kwargs)
        axs[0].add_patch(rect)

        # rect = patches.Rectangle((plot_data.pc-plot_data.ph/2-1, plot_data.xlayers-plot_data.pw/2), plot_data.ph, plot_data.pw-1, **patches_kwargs)
        rect = patches.Rectangle((plot_data.pc-plot_data.ph/2, 0),
                                 plot_data.ph - 1, plot_data.pw/2-1, **patches_kwargs)
        axs[0].add_patch(rect)

        # rect = patches.Rectangle((plot_data.pc-plot_data.ph/2-1-8, plot_data.xlayers-plot_data.pw/2-8), plot_data.ph+16, plot_data.pw-1+16, fill = False, edgecolor = "green")
        # axs[0].add_patch(rect)


def animate_factory(dataframe_animate, pc_list, **kwargs):
    init = True

    def animate(i):
        nonlocal init
        axs[0].cla()
        axs[1].cla()
        dataframe = get_by_kwargs(dataframe_animate, pc=pc_list[i])
        # plot_imshow(dataframe, "phi", colorbar=init, patches_kwargs=dict(color="green"), **kwargs)
        plot_imshow(
            dataframe,
            "phi",
            colorbar=init,
            patches_kwargs=dict(color="green"),
            imshow_kwargs=dict(vmin=0, vmax=0.3),
            **kwargs
        )
        axs[0].set_title(f'Particle position z={pc_list[i]}')
        axs[1].scatter(dataframe_animate.pc, dataframe_animate.free_energy,
                       label="sf-scf", s=2, color="green")
        # axs[1].scatter(dataframe_animate.pc, dataframe_animate.free_energy, label =  "sf-scf", s=2, color = "green")
        axs[1].plot(fe, label="model")
        axs[1].plot(fe_corrected, label="model_gauss")
        axs[1].plot(z_ascf, fe_ascf, label="model_verbose_integration")
        axs[1].scatter(dataframe.pc, dataframe.free_energy, color="green")
        axs[1].text(
            dataframe.pc,
            dataframe.free_energy,
            s=f'{np.round(dataframe.free_energy.squeeze(), 3)}',
            va="bottom",
            ha="right")
        axs[1].legend()
        axs[0].set_ylabel("$r$")
        axs[0].set_xlabel("$z$")
        axs[1].set_ylabel("$F/k_BT$")
        axs[1].set_xlabel("$z$")
        # axs[2].imshow(dataframe.delta_phi_vicinity, origin = "lower", aspect = "equal")
        init = False
    return animate


# %%
# df = pd.read_pickle(f"data/pore_with_particle/r=26_s=52_h=40_l1=120_l2=120_chi_PS={chi_ps}_chi_PW=0_chi_PC=-1_N=300_sigma=0.02_y_c=20_ph=4_pw=4.pkl")
df = sfbox_utils.store.create_master_table(dir="h5")
group_by = [
    'N', 'chi_PC', 'chi_PS', 'chi_PW',
    'h', 'l1', 'l2', 'ph',
    'pw', 'r', 's', 'sigma'
]
utils.ground_energy_correction(df, group_by)

# ground_energy_correction(df, group_columns=["chi_PS", "chi_PC", "ph", "pw"])
# %%
df_to_plot = get_by_kwargs(
    df, chi_PC=chi_pc, chi_PS=chi_ps, s=s, r=r, ph=ph, pw=pw)
utils.load_datasets(df_to_plot, ["phi"])
df_to_plot.sort_values(by="pc", inplace=True)

df_to_plot["delta_phi"] = df_to_plot.phi.apply(lambda x: x - phi_empty)
# df_to_plot["delta_phi_vicinity"] = df_to_plot.apply(lambda _: _.delta_phi[int(_.pc-_.ph/2-8):int(_.pc-_.ph/2+8),int(_.xlayers-_.pw/2-8):int(_.xlayers-_.pw/2+8)], axis = 1)
pc_list = df_to_plot["pc"].to_numpy()
pc_list.sort()
# %%
animate = animate_factory(df_to_plot, pc_list)
# %%
animate(0)
fig.savefig(f"frame_{chi_ps}.pdf")
fig
# %%

ani = animation.FuncAnimation(fig, animate, frames=range(len(pc_list)))
FFwriter = animation.FFMpegWriter()
ani.save(f'animation/plot_{chi_ps}_{chi_pc}.mp4', writer=FFwriter, dpi=300)
# %%
