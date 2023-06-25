#%%
import sfbox_utils
import matplotlib.pyplot as plt
import numpy as np
import pickle
import utils
import pandas as pd
from scipy.signal import convolve
#%%
def cylynder_r0_kernel(radius:int, height:int = None):
    if height is None:
        height = radius*2
    r = np.arange(radius)
    volume_r = np.pi*(2*r+1)
    volume = np.tile(volume_r, (height,1)).T
    surface = np.zeros_like(volume)

    surface[:, 0] = surface[:,-1] = volume_r
    surface[-1,:] =surface[-1,:] + 2*np.pi*radius
    return volume, surface

def cylinder_volume_surface(radius:int, height:int = None):
    if height is None:
        height = radius*2
    V = np.pi*radius**2*height
    S = 2*np.pi*radius**2 + 2*np.pi*radius*height
    return V, S

def Pi(phi, chi_PS, trunc = False):
    Pi_=-np.log(1-phi) - phi - chi_PS*phi**2
    if trunc:
        Pi_[Pi_<1e-16]=0
    return Pi_

def gamma(a0, a1, chi_PS, chi_PC, phi):
    chi_crit = 6*np.log(5/6)
    chi_ads = chi_PC - chi_PS*(1-phi)
    gamma = (chi_ads - chi_crit)*(a0*phi+a1*phi**2)
    return gamma

def free_energy_cylinder(radius, data, a0, a1,  chi_PS, chi_PC):
    volume, surface = cylynder_r0_kernel(radius)
    phi = data["phi"][0:radius]
    phi = np.pad(phi, ((0, 0),(radius,radius-1)))
    Pi_arr = Pi(phi, chi_PS)
    gamma_arr = gamma(a0, a1, chi_PS, chi_PC,  phi)
    osmotic = convolve(Pi_arr, volume, 'valid')[0]
    surface = convolve(gamma_arr, surface, 'valid')[0]
    return osmotic, surface

def free_energy_approx(radius, data, a0, a1, chi_PS, chi_PC):
    volume, surface =cylinder_volume_surface(radius)
    phi = data["phi"][0, :]
    Pi_arr = Pi(phi, chi_PS)
    gamma_arr = gamma(a0, a1, chi_PS, chi_PC,  phi)
    osmotic = Pi_arr*volume
    surface = gamma_arr*surface
    return osmotic, surface


#%%
chi_PC = 0
chi_PS = 1
s = 52
r = 26
ph =8
pw =8
particle_in_pore = pd.read_pickle("reference_table.pkl")
particle_in_pore = utils.get_by_kwargs(
    particle_in_pore,
    chi_PC =chi_PC,
    chi_PS = chi_PS,
    s=s,
    r=r,
    ph=ph,
    pw = pw
    ).sort_values(by = "pc")
#%%
empty_pore = pd.read_pickle("empty_brush.pkl")
empty_pore = utils.get_by_kwargs(empty_pore, chi_PS = chi_PS, r=26, s = 52).squeeze()

a0 = 0.18
a1 = -0.09
osm_appr, sur_appr = free_energy_approx(pw/2, empty_pore,
        a0=a0, a1=a1, chi_PS = chi_PS, chi_PC = chi_PC
    )
tot_appr = osm_appr+sur_appr

osm, sur = free_energy_cylinder(int(pw/2), empty_pore, a0, a1, chi_PS, chi_PC)
tot = osm+sur
#%%
#fig, (ax1, ax2) = plt.subplots(nrows=2, sharex = True)
fig, ax2 = plt.subplots()
#ax1.imshow(empty_pore["phi"], origin = "lower")

ax2.plot(tot_appr, label = "approx")

ax2.plot(tot, label = "conv")

ax2.plot(
    particle_in_pore["pc"],
    particle_in_pore["free_energy"],
    marker = "s",
    linewidth = 0.2,
    markerfacecolor = 'none',
    markeredgecolor = "blue",
    color = "blue",
    label = "sfbox")

ax2.legend()

ax2.set_xlabel("$z$")
ax2.set_ylabel("$\Delta F$")

limits = (min(np.min(particle_in_pore["free_energy"]), np.min(tot)), max(np.max(particle_in_pore["free_energy"]), np.max(tot)))
yscale= limits[1]-limits[0]
ax2.set_ylim(limits[0]-0.2*yscale, limits[1]+0.2*yscale)

ax2.set_xlim(0, particle_in_pore["ylayers"].head(1).squeeze())

ax2.set_title(f"Insertion free energy for {chi_PS=}, {chi_PC=}, {a0=}, {a1=}, d={pw}")
# %%
utils.load_datasets(particle_in_pore, ["phi"])
#%%
r_cut = 50
z_cut = 30
wall_thickness = 52
l1 = 120
extent = [-z_cut-wall_thickness/2, z_cut+wall_thickness/2, -r_cut, r_cut]
def cut_and_mirror(arr):
    cut = arr.T[0:r_cut, l1-z_cut:l1+wall_thickness+z_cut]
    return np.vstack((np.flip(cut), cut[:,::-1]))
for pc, data in particle_in_pore.groupby(by = ["pc"]):
    #zc = pc - int(data["ylayers"]/2)
    zc =int(data["ylayers"]/2)-pc
    if np.abs(zc)>50:continue
    fig, ax = plt.subplots()
    phi = cut_and_mirror(data["phi"].squeeze().T).T
    ax.imshow(phi, cmap = "cividis", origin = "lower", extent = extent)
    levels = [0, 0.001, 0.05, 0.6, 0.68]
    c = ax.contour(phi, colors = "red", origin = "lower", extent = extent, levels = levels)
    ax.clabel(c, inline=True, zorder=2)
    fe = float(data["free_energy"])
    fig.text(0.5, 0.75,
             f"$\Delta F = {fe:.3f}$\n"+"$z_{c}$"+f"$={zc}$", color = "white"
             )

    fig.savefig(f"frames/chi_PS_1.0_d_{pw}/{pc}.pdf")
# %%
r_cut = 50
z_cut = 30
wall_thickness = 52
l1 = 120
extent = [-z_cut-wall_thickness/2, z_cut+wall_thickness/2, -r_cut, r_cut]
def cut_and_mirror(arr):
    cut = arr.T[0:r_cut, l1-z_cut:l1+wall_thickness+z_cut]
    return np.vstack((np.flip(cut), cut[:,::-1]))
phi0 = cut_and_mirror(empty_pore["phi"].squeeze().T).T
for pc, data in particle_in_pore.groupby(by = ["pc"]):
    #zc = pc - int(data["ylayers"]/2)
    zc =int(data["ylayers"]/2)-pc
    if np.abs(zc)>50:continue
    fig, ax = plt.subplots()
    phi = cut_and_mirror(data["phi"].squeeze().T).T
    phi = phi-phi0
    ax.imshow(phi, cmap = "seismic", origin = "lower", extent = extent, vmin = -0.4, vmax = 0.4)
    levels = [0, 0.01, 0.05, 0.1, 0.2, 0.4]
    c = ax.contour(phi, colors = "red", origin = "lower", extent = extent, levels = levels)
    ax.clabel(c, inline=True, zorder=2)
    fe = float(data["free_energy"])
    fig.text(0.5, 0.75,
             f"$\Delta F = {fe:.3f}$\n"+"$z_{c}$"+f"$={zc}$", color = "white"
             )

    fig.savefig(f"frames/dleta_phi_chi_PS_1.0_d_{pw}/{pc}.pdf")
# %%
