#%%
import sys
sys.path.append('..')

from theory import free_energy_phi, D_eff

import pandas as pd
import numpy as np
import ascf_pb

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import functools

#matplotlib settings
LAST_USED_COLOR = lambda: plt.gca().lines[-1].get_color()
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


#%%
N=300
sigma = 0.02
phi_func = ascf_pb.phi(N=N, sigma = sigma)
a1, a2 = [ 0.19814812, -0.08488959]
def _phi_(z, **kwargs):
    z = abs(z)
    return phi_func(z=z, **kwargs)
phi_mirror = np.vectorize(_phi_, excluded = ["N", "sigma", "chi"])

# %%
def plot_phi(chi_PSs):
    fig, ax = plt.subplots()
    for chi_ps in chi_PSs:
        H = ascf_pb.D(N=N, sigma = sigma, chi=chi_ps)()
        z = np.linspace(-H,H)
        phi = phi_mirror(z, chi=chi_ps, N=N, sigma= sigma)
        ax.plot(z, phi, label = chi_ps)
    ax.legend(title = "$\chi_{PS}$")
    ax.set_xlabel("z")
    ax.set_ylabel("phi")
    return fig

chi_PSs = np.round(np.arange(0,1.0,0.1), 3)
fig = plot_phi(chi_PSs)

# %%
N=300
sigma=0.02
a1=0.16
a2=0.08
# %%
b=150
d=10
fig, ax = plt.subplots()
chi_PSs = [0, 0.25, 0.5, 0.6, 0.8]
chi_PCs = np.arange(-3.5, 0.5, 0.2)
for chi_ps in chi_PSs:
    D = []
    for chi_pc in chi_PCs:
        b = ascf_pb.D(N=N, sigma = sigma, chi=chi_ps)()
        D.append(D_eff_membrane(N, sigma, chi_pc, chi_ps, d, b))
    ax.plot(chi_PCs, D, label = chi_ps)
ax.legend(title = "$\chi_{PS}$")
ax.set_xlabel("$\chi_{PC}$")
ax.set_ylabel("$D_{eff}$")
#fig.savefig(f"D_eff_membrane_on_chi_pc_{d}_fixed.pdf")
ax.set_yscale("log")
fig.savefig(f"D_eff_membrane_on_chi_pc_{d}.pdf")
# %%
#b=150
d=4
fig, ax = plt.subplots()
chi_PCs = [0.5, 0, 0.5, -1.0, -2.0, -3.0]
chi_PSs = np.arange(0, 1, 0.05)
for chi_pc in chi_PCs:
    D = []
    for chi_ps in chi_PSs:
        fe_func =fe_membrane_func(N=N, sigma=sigma, chi_PS=chi_ps, a1=a1, a2=a2, chi_PC=chi_pc, d=d)
        b = ascf_pb.D(N=N, sigma = sigma, chi=chi_ps)()
        D.append(D_eff(b, fe_func, half=True))
    ax.plot(chi_PSs, D, label = chi_pc)
ax.legend(title = "$\chi_{PC}$")
ax.set_xlabel("$\chi_{PS}$")
ax.set_ylabel("$D_{eff}$")
ax.set_yscale("log")
#fig.savefig(f"D_eff_membrane_on_chi_ps_{d}.pdf")
# %%
