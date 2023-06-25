# %%
from pystencils.session import *
import numpy as np
from matplotlib import patches
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm
import utils
import pandas as pd
from scipy import ndimage
from functools import lru_cache
from particle_convolution import convolve_particle_surface, convolve_particle_volume

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

np.seterr(divide='ignore')

# %%
def volume(d):
    return np.pi*d**3/6

def surface(d):
    return np.pi*d**2

def gamma(a1, a2, chi_PS, chi_PC, phi):
    chi_crit = 6*np.log(5/6)
    chi_ads = chi_PC - chi_PS*(1-phi)
    gamma = (chi_ads - chi_crit)*(a1*phi+a2*phi**2)
    return gamma

def Pi(phi, chi_PS, trunc = True):
    Pi_=-np.log(1-phi) - phi - chi_PS*phi**2
    if trunc:
        Pi_[Pi_<1e-16]=0
    return Pi_

def surface_free_energy(phi, a1, a2, chi_PS, chi_PC, d):
    return surface(d)*gamma(a1, a2, chi_PS, chi_PC, phi)

def volume_free_energy(phi, chi_PS, d, trunc=True):
    return Pi(phi, chi_PS, trunc)*volume(d)

def free_energy_phi(phi, a1, a2, chi_PS, chi_PC, d):
    return surface_free_energy(phi, a1, a2, chi_PS, chi_PC, d)+volume_free_energy(phi, chi_PS, d)

def mobility_phi(phi, k, d):
    eps = np.where(phi==0, 0.0, 1/phi)
    #print(eps)
    m = eps * eps / (d * d)
    m = m /(1.0 + m**k)**(1 / k)
    m = np.where(phi>0, m, 1.0)
    return m

def read_scf_data(**kwargs):
    empty_pore_all_data = pd.read_pickle("empty_brush.pkl")
    empty_pore = utils.get_by_kwargs(empty_pore_all_data, **kwargs).squeeze()
    empty_pore["phi"] = empty_pore["phi"].T
    return empty_pore

def integrate_cylinder(array_zr):
    Nz, Nr = np.shape(array_zr)
    H = Nz
    A = Nr**2/2
    r = np.arange(0, Nr)
    P_z = np.sum(array_zr*(r+1/2), axis = 1)**(-1)
    P = np.sum(P_z)**(-1)
    return P*H/A

def integrate_partition(array_zr):
    Nz, Nr = np.shape(array_zr)
    H = Nz
    A = Nr**2/2
    r = np.arange(0, Nr)
    P_z = np.sum(array_zr*(r+1/2), axis = 1)
    P = np.sum(P_z)
    return P/(A*H)


def generate_circle_kernel(d):
    radius = d/2
    a = np.zeros((d, d), dtype =bool)
    radius2 = radius**2
    for i in range(d):
        for j in range(d):
            distance2 = (radius-i-0.5)**2 + (radius-j-0.5)**2
            if distance2<radius2:
                a[i,j] = True
    return a

def generate_circle_contour_kernel(d, thickness=2):
    circumference = generate_circle_kernel(d)
    fill = np.zeros_like(circumference)
    fill[thickness:-thickness, thickness:-thickness] = generate_circle_kernel(d-thickness*2)
    circumference = np.logical_and(circumference, np.logical_not(fill))
    return circumference

def read_fields(a0, a1, chi_PC, chi, wall_thickness, pore_radius, dummy = False, d = None, conv = True):
    fields = dict(read_scf_data(chi_PS = chi, s = wall_thickness, r = pore_radius))
    phi = fields["phi"]
    l1 = fields["l1"]
    l2 = fields["l2"]
    if dummy:
        phi=np.zeros_like(phi)

    W_arr = np.zeros_like(phi)
    try:
        W_arr[l1:l1+wall_thickness+1, pore_radius:] = True
    except:
        print(fields)
        print(chi)
        raise Exception

    W_arr = ndimage.binary_dilation(W_arr, structure = generate_circle_kernel(d))
    fields["walls"] = W_arr

    fields["Pi_trunc"] = Pi(phi, chi, True)
    fields["Pi"] = Pi(phi, chi, False)
    fields["gamma"] = gamma(a0, a1, chi, chi_PC, phi)

    if d is not None:
        fields["surface"] = fields["gamma"]*surface(d)
        fields["osmotic"] = fields["Pi"]*volume(d)
        fields["osmotic_trunc"] = fields["Pi_trunc"]*volume(d)

        fields["free_energy"] = fields["surface"] + fields["osmotic"]
        fields["free_energy_trunc"] = fields["surface"] + fields["osmotic_trunc"]

        fields["mobility"] = mobility_phi(phi, 1, d)

        conductivity = fields["mobility"]*np.exp(-fields["free_energy"])
        conductivity[W_arr==True] = 0
        fields["conductivity"] = conductivity

        conductivity = fields["mobility"]*np.exp(-fields["free_energy_trunc"])
        conductivity[W_arr==True] = 0
        fields["conductivity_trunc"] = conductivity

        partition = np.exp(-fields["free_energy"])
        partition[W_arr==True] = 0
        fields["partition"] = partition


        partition = np.exp(-fields["free_energy_trunc"])
        partition[W_arr==True] = 0
        fields["partition_trunc"] = partition


        ###convolved###
        if conv:
            fields["osmotic_conv"] = convolve_particle_volume(fields["Pi"].T, d).T
            #fields["osmotic_trunc_conv"] = convolve_particle_volume(fields["Pi_trunc"].T, d).T

            fields["surface_conv"] = convolve_particle_surface(fields["gamma"].T, d).T

            fields["free_energy_conv"] = fields["surface_conv"] + fields["osmotic_conv"]
            #fields["free_energy_trunc_conv"] = fields["surface_conv"] + fields["osmotic_trunc_conv"]

            conductivity = fields["mobility"]*np.exp(-fields["free_energy_conv"])
            conductivity[W_arr==True] = 0
            fields["conductivity_conv"] = conductivity

            partition = np.exp(-fields["free_energy_conv"])
            partition[W_arr==True] = 0
            fields["partition_conv"] = partition

    return fields


@lru_cache
def pore_permeability(a0, a1, d, chi_PC, chi, wall_thickness, pore_radius, l, dummy = False, trunc = False, conv = False):
    fields = read_fields(a0, a1, chi_PC, chi, wall_thickness, pore_radius, dummy, d, conv = conv)
    if trunc:
        conductivity = fields["conductivity_trunc"]
        empty_pore_conductivity = read_fields(a0, a1, chi_PC, chi, wall_thickness, pore_radius, dummy=True, d=d, conv = conv)["conductivity_trunc"]
    else:
        if conv:
            conductivity = fields["conductivity_conv"]
        else:
            conductivity = fields["conductivity"]

        empty_pore_conductivity = read_fields(a0, a1, chi_PC, chi, wall_thickness, pore_radius, dummy=True, d=d, conv = False)["conductivity"]

    l1 = fields["l1"]
    wall_thickness = fields["s"]
    pore_radius = fields["r"]

    conductivity = conductivity[l1-l:l1+wall_thickness+l, 0:pore_radius]
    empty_pore_conductivity = empty_pore_conductivity[l1-l:l1+wall_thickness+l, 0:pore_radius]

    P_empty = integrate_cylinder(empty_pore_conductivity)
    if P_empty == 0: return 0
    P = integrate_cylinder(conductivity)
    return P/P_empty

@lru_cache
def pore_partition(a0, a1, d, chi_PC, chi, wall_thickness, pore_radius, l, dummy = False, trunc = False, conv = False):
    fields = read_fields(a0, a1, chi_PC, chi, wall_thickness, pore_radius, dummy, d, conv=conv)
    if trunc:
        partition = fields["partition_trunc"]
    else:
        if conv:
            partition = fields["partition_conv"]
        else:
            partition = fields["partition"]
    l1 = fields["l1"]
    wall_thickness = fields["s"]
    pore_radius = fields["r"]
    partition = partition[l1-l:l1+wall_thickness+l, 0:pore_radius]
    PC = integrate_partition(partition)
    return PC

#%%

# %%
#%%
a0 = 0.18
a1 = -0.09
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness
#%%
d = 24
chi_PC = 0
chi = 0.9

pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness

fields = read_fields(
    a0, a1, d=d,
    chi_PC=chi_PC, chi=chi,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    )
phi = fields["phi"]
W_arr = fields["walls"]
fe = fields["free_energy"]
mobility = fields["mobility"]
conductivity = fields["conductivity"]
l1 = fields["l1"]

fields["log(conductivity)"]=np.clip(np.log(fields["conductivity"]), -20, 20)
fields["log(conductivity_trunc)"]=np.clip(np.log(fields["conductivity_trunc"]), -20, 20)
fields["log(partition)"]=np.clip(np.log(fields["partition"]), -20, 20)
fields["log(partition_trunc)"]=np.clip(np.log(fields["partition_trunc"]), -20, 20)

fields["log(conductivity_conv)"]=np.clip(np.log(fields["conductivity_conv"]), -20, 20)
fields["log(partition_conv)"]=np.clip(np.log(fields["partition_conv"]), -20, 20)


#fields["free_energy"][W_arr==True]
#%%
#%matplotlib ipympl
from heatmap_explorer import plot_heatmap_and_profiles

r_cut = 50
z_cut = 30
def cut_and_mirror(arr):
    cut = arr.T[0:r_cut, l1-z_cut:l1+wall_thickness+z_cut]
    return np.vstack((np.flip(cut), cut[:,::-1]))
extent = [-z_cut-wall_thickness/2, z_cut+wall_thickness/2, -r_cut, r_cut]



#for key in ["phi", "Pi", "gamma", "mobility", "free_energy", "free_energy_trunc", "log(conductivity)"]:
for key in ["osmotic", "osmotic_conv", "surface", "surface_conv", "free_energy_trunc", "free_energy_conv", "log(conductivity_trunc)", "log(conductivity_conv)"]:
    fig = plot_heatmap_and_profiles(
        cut_and_mirror(fields[key]),
        x0=-50,
        y0=-30-wall_thickness/2,
        xlabel="$r$",
        ylabel = "$z$",
        zlabel=key,
        update_zlim=False,
        hline_y=int(z_cut+wall_thickness/2),
        vline_x=r_cut
        #zmin="0.01",
        #zmax = "0.98"
        )
    fig.show()

#%%
D = [4, 8, 16, 32]
CHI_PC = [0,-0.5,-1,-1.5]
CHI_PS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
#CHI_PS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
convolve = True
trunc = False
#L = 0
L=52
P = [[[
    pore_permeability(
        a0, a1, d_, chi_pc_, chi_ps_, wall_thickness, pore_radius, L,
        #dummy = True,
        trunc = trunc,
        conv = convolve
        )
    for chi_pc_ in CHI_PC
    ] for chi_ps_ in CHI_PS
    ] for d_ in D
    ]

fig, axs = plt.subplots(ncols = 2, nrows = 2, sharex = True)
for ax, d_, p_ in zip(axs.flatten(), D, P):
    for chi_pc_, p__ in zip(CHI_PC, np.array(p_).T):
        ax.plot(CHI_PS, p__, label = chi_pc_, marker = "o", markerfacecolor = "none")
        ax.set_title(f"d={d_}")
        ax.set_yscale("log")
        ax.set_xlabel("$\chi_{PS}$")
        ax.set_ylabel("$P/P_0$")
        ax.set_ylim(1e-6, 1e2)
        ax.axhline(y = 1, color = "black")

ax.legend(title = "$\chi_{PC}$")
#fig.set_size_inches(5,4)
fig.suptitle(f"Permeability for {L=}")
#ax.set_ylim(4e-2, 2e2)
fig.savefig(f"fig/permeability/Permeation_on_chi_PS_{L=}_{trunc=}_{convolve=}.pdf")
#fig.savefig(f"fig/permeability/Permeation_on_chi_PS_{L=}_no_vol_excl.pdf")
 #%%
#D = np.arange(2, 32)
D = [2, 4, 8, 16, 24, 32, 40]
CHI_PC = [0, -0.5, -0.6, -0.7, -0.8]
CHI_PS = [0.6, 0.7, 0.8]
convolve = False
trunc = True
L = 0
P = [[[
    pore_permeability(
        a0, a1, d_, chi_pc_, chi_ps_, wall_thickness, pore_radius, L,
        trunc  = trunc,
        conv = convolve
        )
    for d_ in D
    ]
    for chi_pc_ in CHI_PC
    ]
    for chi_ps_ in CHI_PS
    ]
#%%
fig, axs= plt.subplots(nrows =1, ncols = 3)
for ax, chi_ps_, p_ in zip(axs.flatten(), CHI_PS, P):
    for chi_pc_, p__ in zip(CHI_PC, p_):
        ax.plot(D, p__, label = chi_pc_, marker = "o", markerfacecolor = "none")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("$\chi_{PS}=$"+f"{chi_ps_}")
    ax.set_xlabel("$d$")
    ax.set_ylabel("$P/P_0$")
    ax.set_ylim(1e-10, 1e10)
    ax.axhline(y = 1, color = "black")
ax.legend(title = "$\chi_{PC}$")
#fig.set_size_inches(5,4)
fig.suptitle(f"Permeability on d, {L=} {trunc=} {convolve=}")
#ax.set_ylim(4e-2, 2e2)
plt.tight_layout()
fig.savefig(f"fig/permeability/Permeation_on_d_{L=}_{trunc=}_{convolve=}.pdf")
#%%
#D = np.arange(2, 32)
D = [2, 3, 4, 6, 8, 12, 16, 24, 32]
CHI_PC = [0,-0.5,-1,-1.5, -2]
CHI_PS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
L = 52
P = [[[
    pore_partition(
        a0, a1, d_, chi_pc_, chi_ps_, wall_thickness, pore_radius, L,
        trunc = True,
        convolve=convolve
        )
    for d_ in D
    ]
    for chi_pc_ in CHI_PC
    ]
    for chi_ps_ in CHI_PS
    ]
#%%

fig, axs= plt.subplots(nrows =2, ncols = 3)
for ax, chi_ps_, p_ in zip(axs.flatten(), CHI_PS, P):
    for chi_pc_, p__ in zip(CHI_PC, p_):
        ax.plot(D, p__, label = chi_pc_, marker = "o", markerfacecolor = "none")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("$\chi_{PS}=$"+f"{chi_ps_}")
    ax.set_xlabel("$d$")
    ax.set_ylabel("$PC$")
    ax.set_ylim(1e-10, 1e10)
    ax.axhline(y = 1, color = "black")
ax.legend(title = "$\chi_{PC}$")
#fig.set_size_inches(5,4)
fig.suptitle(f"PC on d, {L=} {trunc=} {convolve=}")
#ax.set_ylim(4e-2, 2e2)
plt.tight_layout()
fig.savefig(f"fig/permeability/PC_on_d_{L=}_{trunc=}_{convolve=}.pdf")
#%%
#fig, axs = plt.subplots(nrows = 2, ncols=2, sharex=True, sharey = True)
#plt.subplots_adjust(wspace=-0.6, hspace = 0.4)
#
#def cut_and_mirror(arr):
#    cut = arr.T[0:50, l1-30:l1+wall_thickness+30]
#    return np.vstack((np.flip(cut), cut))
#
##current_cmap = matplotlib.cm.get_cmap()
##current_cmap.set_colors = "seismic"
##current_cmap.set_bad(color='lightgrey')
#
#extent = [-30-wall_thickness/2, 30+wall_thickness/2, -50, 50]
#for ax, field, title in zip(
#            axs.flatten(),
#            [phi, fe, mobility, conductivity],
#            ["$\phi$", "$\Delta F$", "$D/D_0$", r"$\log(\frac{D}{D_0} e^{-\Delta F})$"]
#            ):
#    field_ = cut_and_mirror(field)
#    if field is fe:
#        im = ax.imshow(
#            field_, origin = "lower",
#            #vmin = np.percentile(fe, 0.01), vmax = np.percentile(fe, 0.99),
#            extent = extent,
#            aspect = "equal",
#            cmap = "seismic",
#            interpolation = "nearest"
#            )
#
#    elif field is conductivity:
#        im = ax.imshow(
#            #field_,
#            np.log(field_),
#            origin = "lower",
#            extent = extent,
#            aspect = "equal",
#            cmap = "seismic",
#            interpolation = "nearest"
#            )
#    else:
#        im = ax.imshow(
#            field_,
#            origin = "lower",
#            extent = extent,
#            aspect = "equal",
#            cmap = "seismic",
#            interpolation = "nearest"
#            )
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    plt.colorbar(im, cax=cax)
#    ax.add_patch(patches.Rectangle((-wall_thickness/2, pore_radius), wall_thickness, 25, color = "green", zorder=3))
#    ax.add_patch(patches.Rectangle((-wall_thickness/2, -pore_radius), wall_thickness, -25, color = "green", zorder=3))
#    ax.set_title(title)
#
#field_ = cut_and_mirror(W_arr)
#ax.contour(field_, extent = extent, levels = [1], colors = "green", interpolation = "nearest")
#
#fig.supxlabel("z")
#fig.suptitle(f"Fields for {chi=}, {chi_PC=}, {d=}")
#
#fig.savefig(f"fig/permeability/fields_{chi=}_{chi_PC=}_{d=}.pdf", bbox_inches = "tight")
##%%
#chi_PC = -1
#d=32
#CHI_PS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
#fig, ax = plt.subplots()
#L=wall_thickness
#z = np.arange(-(wall_thickness)/2+0.5-L,+wall_thickness/2+0.5+L)
#for chi_ps in CHI_PS:
#    conductivity_z = pore_conductivity_z(a0, a1, d, chi_PC, chi_ps, wall_thickness, pore_radius, L)
#    ax.plot(z, conductivity_z, label = chi_ps)
#ax.axvline(x = -(wall_thickness)/2, color = "black", linewidth = 2)
#ax.axvline(x = (wall_thickness)/2, color = "black", linewidth = 2)
#ax.set_yscale("log")
#ax.set_xlabel("$z$")
#ax.set_ylabel("$\int^R D e^{- \Delta F} r dr$")
#ax.legend(title = "$\chi_{PS}$")
#
## %%
#chi_PC = -1
#d=32
#CHI_PS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
#fig, ax = plt.subplots()
#L=wall_thickness
#r = np.arange(pore_radius)
#
#
#
#for chi_ps in CHI_PS:
#    fields = read_fields(
#    a0, a1, d=d,
#    chi_PC=chi_PC, chi=chi_ps,
#    wall_thickness=wall_thickness,
#    pore_radius=pore_radius,
#    )
#    l1 = fields["l1"]
#    fe = fields["free_energy"]
#    fe_r = fe[int(l1+wall_thickness/2), :pore_radius]
#    ax.plot(r, fe_r, label = chi_ps)
#
#ax.legend()
## %%
#CHI_PS = [0.6]
#fig, (ax, ax2) = plt.subplots(nrows =2)
#for chi_ps in CHI_PS:
#    fields = read_fields(
#    a0, a1, d=d,
#    chi_PC=chi_PC, chi=chi_ps,
#    wall_thickness=wall_thickness,
#    pore_radius=pore_radius,
#    )
#    r = np.arange(pore_radius)
#    l1 = fields["l1"]
#    phi = fields["phi"]
#    phi_r = phi[int(l1+wall_thickness/2), :pore_radius]
#
#    osmotic = Pi(phi_r, chi_ps)
#    surf = gamma(a0, a1, chi_ps, chi_PC, phi_r)
#
#    #fe = fields["free_energy"]
#    #fe_r = fe[int(l1+wall_thickness/2), :pore_radius]
#
#    #ax.plot(r, fe_r, label = "total")
#    ax.plot(r, osmotic, label = "Pi")
#    ax.plot(r, surf, label = "gamma")
#
#    ax2.plot(r, phi_r)
#
#ax.legend()
## %%
#import scf_pb
##H = scf_pb.D(N=300, sigma=0.02, chi=0.8)
#phi_D =scf_pb.phi_v(N=300, sigma=0.02, chi=0.6, z = ["H"])
#phi_D
# %%
