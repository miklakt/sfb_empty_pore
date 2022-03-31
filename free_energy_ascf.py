#%%
from cProfile import label
import ascf_pb
import numpy as np
import functools
import matplotlib.pyplot as plt
from sympy import arg
from diffusion import D_eff

a1 = 0.16
a2 = 0.08

def volume(w, h = None):
    if h is None:
        h=w
    return np.pi*w**2/4*h

def surface(w, h = None):
    if h is None:
        h=w
    return np.pi*w*h+np.pi*w**2/2

def gamma(a1, a2, chi_PS, chi_PC, phi):
    chi_crit = 6*np.log(5/6)
    chi_ads = chi_PC - chi_PS*(1-phi)
    gamma = (chi_ads - chi_crit)*(a1*phi+a2*phi**2)
    return gamma

def Pi(phi, chi_PS):
    return -np.log(1-phi) - phi - chi_PS*phi**2

def surface_free_energy(phi, a1, a2, chi_PS, chi_PC, w, h=None):
    return surface(w, h)*gamma(a1, a2, chi_PS, chi_PC, phi)

def volume_free_energy(phi, chi_PS, w, h=None):
    return Pi(phi, chi_PS)*volume(w,h)


@functools.lru_cache()
def free_energy_penalty_phi(phi, a1, a2, chi_PS, chi_PC, w, h=None):
    return surface_free_energy(phi, a1, a2, chi_PS, chi_PC, w, h)+volume_free_energy(phi, chi_PS, w, h)

@functools.lru_cache()
def free_energy_penalty_plain(z, N, sigma, a1, a2, chi_PS, chi_PC, w, h=None):
    phi = ascf_pb.phi(N=N, sigma = sigma, chi = chi_PS)(z)
    return free_energy_penalty_phi(phi, a1, a2, chi_PS, chi_PC, w, h)

@functools.lru_cache
def D_effective_membrane(N, sigma, a1, a2, chi_PS, chi_PC, w, h=None):
    H = ascf_pb.D(N=N, sigma=sigma, chi = chi_PS)()
    U = functools.partial(
        free_energy_penalty_plain,
        N=N, sigma=sigma, 
        a1=a1, a2=a2, 
        chi_PS=chi_PS, chi_PC=chi_PC, 
        w=w, h=h
    )
    D = D_eff(b=H, U_x=U, half = True, parity='even')
    return D



#%%
N=300
sigma = 0.02
chi_PSs = [0, 0.5, 0.6, 0.8]
fig, ax = plt.subplots()
for chi_ps in chi_PSs:
    D = ascf_pb.D(N=N, sigma=sigma, chi = chi_ps)()
    z = np.linspace(0,D)
    phi_func = np.vectorize(ascf_pb.phi(N=N, sigma=sigma, chi = chi_ps))
    phi = list(phi_func(z))
    phi.append(0)
    z=list(z)
    z.append(z[-1])
    ax.plot(z,phi, label = chi_ps)
    ax.set_xlabel("$z$")
    ax.set_ylabel("$\phi$")
ax.legend(title = "$\chi_{PS}$")
plt.tight_layout()
plt.savefig("phi_plain.pdf")
# %%
w = 10
chi_PCs = [0.5, -0.0, -0.75, -1.5, -2.5]
chi_PSs=[0, 0.5, 0.6, 0.8]
fig, ax = plt.subplots(ncols=len(chi_PSs), sharey=True)
for i, chi_ps in enumerate(chi_PSs):
    for chi_pc in chi_PCs:
        D = ascf_pb.D(N=N, sigma=sigma, chi = chi_ps)()
        z = np.linspace(0,D)
        z=list(z)
        z.append(z[-1])
        free_energy = [free_energy_penalty_plain(z_, N, sigma, a1, a2, chi_ps, chi_pc, w) for z_ in z]
        free_energy[-1]=0
        ax[i].plot(z,free_energy, label = chi_pc)
    ax[i].axhline(y= 0, color = "black")
    ax[i].set_xlabel("z")
    ax[i].set_title("$\chi_{PS}=$"+f"{chi_ps}")
ax[0].set_ylabel("$F/k_BT$")
ax[0].legend(title = "$\chi_{PC}$")
fig.set_size_inches(7,4)
fig.suptitle(f"$N={N}, \sigma = {sigma}, d={w}$")
plt.tight_layout()
plt.savefig("free_energy.pdf")
# %%
chi_PCs = np.linspace(0.5, -3.0)
chi_PSs = [0, 0.5, 0.6, 0.8]
fig, ax = plt.subplots()
for chi_ps in chi_PSs:
    D_eff_chi = [D_effective_membrane(N, sigma, a1, a2, chi_ps, chi_pc, w) for chi_pc in chi_PCs]
    ax.plot(chi_PCs,D_eff_chi, label = chi_ps)
    ax.set_xlabel("$\chi_{PC}$")
    ax.set_ylabel("$D_{eff}$")
    ax.set_yscale("log")
ax.legend(title = "$\chi_{PS}$")
plt.tight_layout()
plt.savefig("D_eff_membrane.pdf")
# %%
chi_PCs = [0.5, 0, -0.5, -1.0, -1.5, -2.0, -2.5]
chi_PSs = np.linspace(0, 0.8)
fig, ax = plt.subplots()
for chi_pc in chi_PCs:
    D_eff_chi = [D_effective_membrane(N, sigma, a1, a2, chi_ps, chi_pc, w) for chi_ps in chi_PSs]
    ax.plot(chi_PSs,D_eff_chi, label = chi_pc)
    ax.set_xlabel("$\chi_{PS}$")
    ax.set_ylabel("$D_{eff}$")
    ax.set_yscale("log")
ax.legend(title = "$\chi_{PC}$")
plt.tight_layout()
plt.savefig("D_eff_membrane_2.pdf")
# %%
