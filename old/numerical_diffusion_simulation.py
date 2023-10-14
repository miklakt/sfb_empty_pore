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

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

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

def Pi(phi, chi_PS):
    Pi_=-np.log(1-phi) - phi - chi_PS*phi**2
    Pi_[Pi_<0]=0
    return Pi_

def surface_free_energy(phi, a1, a2, chi_PS, chi_PC, d):
    return surface(d)*gamma(a1, a2, chi_PS, chi_PC, phi)

def volume_free_energy(phi, chi_PS, d):
    return Pi(phi, chi_PS)*volume(d)

def free_energy_phi(phi, a1, a2, chi_PS, chi_PC, d):
    return surface_free_energy(phi, a1, a2, chi_PS, chi_PC, d)+volume_free_energy(phi, chi_PS, d)

def mobility_phi(phi, k, d):
    eps = np.where(phi==0, 0.0, 1/phi)
    print(eps)
    m = eps * eps / (d * d)
    m = m /(1.0 + m**k)**(1 / k)
    m = np.where(phi>0, m, 1.0)
    return m
def read_scf_data(**kwargs):
    empty_pore_all_data = pd.read_pickle("empty_brush.pkl")
    empty_pore = utils.get_by_kwargs(empty_pore_all_data, **kwargs).squeeze()
    phi = empty_pore["phi"].T
    return phi
def pad_scf_array(array, new_domain):
    nz, nr = new_domain
    nz_old, nr_old = np.shape(array)
    pad_left = pad_right = int((nz - nz_old)/2)
    pad_bottom = 0
    pad_top = nr - nr_old
    return np.pad(array, ((pad_left, pad_right), (pad_bottom, pad_top)), "edge")

def integrate_cylinder(array_zr):
    Nz, Nr = np.shape(array_zr)
    H = Nz
    A = Nr**2/2
    r = np.arange(0, Nr)
    P_z = np.sum(array_zr*(r+1/2), axis = 1)**(-1)
    P = np.sum(P_z)**(-1)
    return P*H/A

get_pore = lambda _: _[l1:l1+wall_thickness, 0:pore_radius]
get_pore_2 = lambda _, l: _[l1-l:l1+wall_thickness+l, 0:pore_radius]

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

@lru_cache
def pore_permeability(a0, a1, d, chi_PC, chi, l):
    W_arr = np.zeros(domain_size, dtype = np.int8)
    W_arr[l1:l1+wall_thickness+1, pore_radius:] = True
    W_arr = ndimage.binary_dilation(W_arr, structure = generate_circle_kernel(d))
    phi = read_scf_data(chi_PS = chi, s = wall_thickness, r = pore_radius)
    phi = pad_scf_array(phi, domain_size)
    fe = free_energy_phi(phi, a0, a1, chi, chi_PC, d)
    fe = fe + W_arr*1e100
    mobility = mobility_phi(phi, 1, d)
    conductivity = mobility*np.exp(-fe)
    conductivity = get_pore_2(conductivity, l)
    return integrate_cylinder(conductivity)
# %%
a0 = 0.18
a1 = -0.09
d = 32
chi_PC = -3
chi = 0.9

h = 200 # r+h total radius of the system
l1 = 200 # distance before the wall
l2 = 200 # distance after the wall
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness

z = l1+l2+wall_thickness 
r = pore_radius+h
domain_size = (z, r)

#mask for walls
W_arr = np.zeros(domain_size, dtype = np.int8)
W_arr[l1:l1+wall_thickness+1, pore_radius:] = True
W_arr = ndimage.binary_dilation(W_arr, structure = generate_circle_kernel(d))
#%%
empty_pore =False

if not empty_pore:
    phi = read_scf_data(chi_PS = chi, s = wall_thickness, r = pore_radius)
    phi = pad_scf_array(phi, domain_size)
    fe = free_energy_phi(phi, a0, a1, chi, chi_PC, d)
    fe_min = np.min(fe)
    fe_max = np.max(fe)
    #fe = fe + W_arr*1e100
    mobility = mobility_phi(phi, 1, d)
    #mobility = np.ones_like(fe)
    conductivity = mobility*np.exp(-fe)

    fe[W_arr==True]=np.nan
    mobility[W_arr==True]=np.nan
    conductivity[W_arr==True]=np.nan

    permeability = integrate_cylinder(get_pore(conductivity))
else:
    permeability = 1.0
#%%
fig, axs = plt.subplots(nrows = 2, ncols=2, sharex=True, sharey = True)
plt.subplots_adjust(wspace=-0.6, hspace = 0.4)

def cut_and_mirror(arr):
    cut = arr.T[0:50, l1-30:l1+wall_thickness+30]
    return np.vstack((np.flip(cut), cut))

current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color='lightgrey')

extent = [-30-wall_thickness/2, 30+wall_thickness/2, -50, 50]
for ax, field, title in zip(
            axs.flatten(), 
            [phi, fe, mobility, conductivity], 
            ["$\phi$", "$\Delta F$", "$D/D_0$", r"$\log(\frac{D}{D_0} e^{-\Delta F})$"]
            ):
    field_ = cut_and_mirror(field)
    if field is fe:
        im = ax.imshow(
            field_, origin = "lower", 
            vmin = fe_min, vmax = fe_max, 
            extent = extent, 
            aspect = "equal"
            )

    elif field is conductivity:
        im = ax.imshow(
            np.log(field_), origin = "lower", 
            extent = extent,
            aspect = "equal"
            )
    else:
        im = ax.imshow(
            field_, 
            origin = "lower", 
            extent = extent,
            aspect = "equal"
            )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.add_patch(patches.Rectangle((-wall_thickness/2, pore_radius), wall_thickness, 50, color = "grey", zorder=3))
    ax.add_patch(patches.Rectangle((-wall_thickness/2, -pore_radius), wall_thickness, -50, color = "grey", zorder=3))
    ax.set_title(title)

fig.supxlabel("z")
fig.suptitle(f"Fields for {chi=}, {chi_PC=}, {d=}")

fig.savefig(f"fig/permeability/fields_{chi=}_{chi_PC=}_{d=}.pdf", bbox_inches = "tight")
#fig.savefig(f"fig/permeability/fields_{chi=}_{chi_PC=}_{d=}_no_vol_excl.pdf", bbox_inches = "tight")
#%%
#CHI_PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
#for chi_ps in chi_PS:


#%%
D = [4, 8, 16, 32]
CHI_PC = [0,-1,-2,-3]
CHI_PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
#L = wall_thickness
L=-10
P = [[[
    pore_permeability(a0, a1, d_, chi_pc_, chi_ps_, L) 
    for chi_pc_ in CHI_PC
    ] for chi_ps_ in CHI_PS
    ] for d_ in D
    ]
#%%
fig, axs = plt.subplots(ncols = 2, nrows = 2, sharex = True)
for ax, d_, p_ in zip(axs.flatten(), D, P):
    for chi_pc_, p__ in zip(CHI_PC, np.array(p_).T):
        ax.plot(CHI_PS, p__, label = chi_pc_, marker = "o", markerfacecolor = "none")
        ax.set_title(f"d={d_}")
        ax.set_yscale("log")
        ax.set_xlabel("$\chi_{PS}$")
        ax.set_ylabel("$P/D_0$")
        #ax.set_ylim(1e-5, 1e3)
        ax.axhline(y = 1, color = "black")

ax.legend(title = "$\chi_{PC}$")
#fig.set_size_inches(5,4)
fig.suptitle(f"Permeability for {L=}")
#ax.set_ylim(4e-2, 2e2)
fig.savefig(f"fig/permeability/Permeation_on_chi_PS_{L=}.pdf")
#fig.savefig(f"fig/permeability/Permeation_on_chi_PS_{L=}_no_vol_excl.pdf")
 #%%
D = np.arange(2, 32)
CHI_PC = [0,-1,-2,-3]
CHI_PS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
#L = wall_thickness
L = 0
P = [[[
    pore_permeability(a0, a1, d_, chi_pc_, chi_ps_, L)
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
    ax.set_ylabel("$P/D_0$")
    ax.set_ylim(1e-4, 1e2)
    ax.axhline(y = 1, color = "black")
ax.legend(title = "$\chi_{PC}$")
#fig.set_size_inches(5,4)
fig.suptitle(f"Permeability for on d, {L=}")
#ax.set_ylim(4e-2, 2e2)
plt.tight_layout()
fig.savefig(f"fig/permeability/Permeation_on_d_{L=}.pdf")
#fig.savefig(f"fig/permeability/Permeation_on_d_{L=}_no_vol_excl.pdf")
 # %%
# create arrays
c_arr = np.zeros(domain_size)
U_arr = np.zeros(domain_size)
D_arr = np.ones(domain_size)
J_arr = np.zeros((z,r,2))
W_arr = np.zeros(domain_size, dtype = np.int8)

#assign values
W_arr[l1:l1+wall_thickness+1, pore_radius:] = True
W_arr = ndimage.binary_dilation(W_arr, structure = generate_circle_kernel(d))
if not empty_pore:
    np.copyto(U_arr, fe)
    np.copyto(D_arr, mobility)
    try:
        c_arr = np.loadtxt(f"numerical_diffusion_data/c_arr_{chi=}_{chi_PC=}_{d=}.txt")
    except:
        print("No previous calculation is found")
else:
    try:
        c_arr = np.loadtxt(f"numerical_diffusion_data/c_arr_empty.txt")
    except:
        print("No previous calculation is found")


# create fields
c, c_next, D = ps.fields("c, c_next, D: [2d]", c=c_arr, c_next=c_arr, D = D_arr)
U = ps.fields("U: [2d]", U=U_arr)
J = ps.fields("J(2): [2d]", J = J_arr)
W = ps.fields("W: int[2d]", W=W_arr)

# cylindrical coordinates curvature
ρ_arr = np.ones(domain_size)
ρ_arr[:] = np.arange(0, r)
ρ = ps.fields("ρ: [2d]", ρ = ρ_arr)
# %%
@ps.kernel
def concentration_kernel_cylindrical_desc():
    dt=0.02 #timestep

    c_C = c[0,0]
    c_E = c[1,0]
    c_W = c[-1,0]
    c_N = c[0,1]
    c_S = c[0,-1]

    D_C = D[0,0]
    D_E = D[1,0]
    D_W = D[-1,0]
    D_N = D[0,1]
    D_S = D[0,-1]

    U_C = U[0,0]
    U_E = U[1,0]
    U_W = U[-1,0]
    U_N = U[0,1]
    U_S = U[0,-1]

    c_e = (c_E+c_C)/2
    c_w = (c_W+c_C)/2
    c_n = (c_N+c_C)/2
    c_s = (c_S+c_C)/2

    D_e = (D_E+D_C)/2
    D_w = (D_W+D_C)/2
    D_n = (D_N+D_C)/2
    D_s = (D_S+D_C)/2

    W_C = W[0,0]
    W_E = W[1,0]
    W_W = W[-1,0]
    W_N = W[0,1]
    W_S = W[0,-1]

    J_Eu = D_e*c_e*(U_E-U_C)
    J_Wu = D_w*c_w*(U_W-U_C)
    J_Nu = D_n*c_n*(U_N-U_C)
    J_Su = D_s*c_s*(U_S-U_C)

    J_Ed = D_e*(c_E-c_C)
    J_Wd = D_w*(c_W-c_C)
    J_Nd = D_n*(c_N-c_C)
    J_Sd = D_s*(c_S-c_C)

    J_E = sp.Piecewise((0.0, (W_E+W_C)>0), (J_Ed+J_Eu, True))
    J_W = sp.Piecewise((0.0, (W_W+W_C)>0), (J_Wd+J_Wu, True))
    J_N = sp.Piecewise((0.0, (W_N+W_C)>0), (J_Nd+J_Nu, True))
    J_S = sp.Piecewise((0.0, (W_S+W_C)>0), (J_Sd+J_Su, True))

    J[0,0][0] @= -J_E
    J[0,0][1] @= -J_N

    J_tot = J_E + J_W + (J_N*ρ[0,1] +J_S*ρ[0,0]) / (ρ[0,1]**2/2 - ρ[0,0]**2/2)

    c_next[0,0] @= c[0,0]+J_tot*dt


gl_spec = [(1, 1),(0, 1)]  # no ghost layer at the bottom boundary
ast = ps.create_kernel(
    concentration_kernel_cylindrical_desc,
    cpu_openmp=True,
    ghost_layers=gl_spec,
    )
concentration_kernel = ast.compile()

# %%
radius_factor_source = 4
k = generate_circle_contour_kernel(pore_radius*radius_factor_source*2)
source_boundary=np.zeros_like(c_arr)
source_boundary[l1-pore_radius*radius_factor_source:l1,0:pore_radius*radius_factor_source] = np.array(k[:-pore_radius*radius_factor_source,pore_radius*radius_factor_source:])

radius_factor_sink = 2
k = generate_circle_contour_kernel(pore_radius*radius_factor_sink*2)
sink_boundary=np.zeros_like(c_arr)

sink_boundary[l1+wall_thickness+1:l1+wall_thickness+1+pore_radius*radius_factor_sink,0:pore_radius*radius_factor_sink] = np.array(k[pore_radius*radius_factor_sink:,pore_radius*radius_factor_sink:])
def stationarity_metric(arr0):
    arr =np.zeros_like(arr0)
    np.copyto(arr, arr0)
    def f(arr1):
        return np.max((arr-arr1)**2)
    return f

def boundary_handling(c, J):
    # Left z=0
    c[0, :] = 1
    # Down rho=0
    #c[:, 0] = c[:, 1]
    # Up rho=rho_max
    c[:, -1] = c[:, -2]
    # Right z = z_max
    c[-1, :] = 0

    c[sink_boundary==True] = 0
    
c_tmp_arr = np.empty_like(c_arr)

def timeloop(steps=5000):
    global c_arr, c_tmp_arr, D_arr,  ρ_arr, c_arr, c_tmp_arr
    score_f = stationarity_metric(c_arr) 
    for i in range(steps):
        boundary_handling(c_arr, J_arr)
        concentration_kernel(
            c=c_arr, c_next=c_tmp_arr, 
            D=D_arr, U=U_arr, 
            ρ = ρ_arr, J = J_arr, 
            W = W_arr
        )
        c_arr, c_tmp_arr = c_tmp_arr, c_arr
    score = score_f(c_tmp_arr)
    np.savetxt(f"tmp_data.txt", c_arr)
    print(score)
    return score

#%%
fig, ax  = plt.subplots()
ax.imshow(sink_boundary.T, origin = "lower")
ax.add_patch(patches.Rectangle((l1-1, r-h-1), wall_thickness+2, h+2, color = "white", zorder=3))
#%%
score = 1000
while score > 1e-4:
    score = timeloop()
#%%
timeloop(500000)
fig, ax  = plt.subplots()
ax.imshow(c_arr.T, origin = "lower", vmin = 0, vmax= 2.0)
ax.add_patch(patches.Rectangle((l1-1, r-h-1), wall_thickness+2, h+2, color = "grey", zorder=3))
# %%
fig = ps.plot.vector_field(J_arr, step = 10).figure
ax = fig.gca()
fig.set_size_inches(7,4)
ax.set_xlim(0, np.shape(J_arr)[0]/10)
ax.set_ylim(0, np.shape(J_arr)[1]/10)
J_z_total = np.mean(np.sum(J_arr[1:-1,:,0]*(ρ_arr[1:-1, :]+1/2), axis = 1))
ax.set_title(f"Flux for {chi=}, {chi_PC=}, {d=}, {J_z_total=:.3f}")
if empty_pore:
    fig.savefig(f"fig/permeability/quiver_empty.pdf")
else:
    fig.savefig(f"fig/permeability/quiver_{chi=}_{chi_PC=}_{d=}.pdf")
#%%
print("should be flat for stationary process")
plt.plot(np.sum(J_arr[1:-1,:,0]*(ρ_arr[1:-1, :]+1/2), axis = 1))
#%%
ps.plot.vector_field_magnitude(J_arr)
# %%
if empty_pore:
    np.savetxt(f"numerical_diffusion_data/c_arr_empty.txt", c_arr)
else:
    np.savetxt(f"numerical_diffusion_data/c_arr_{chi=}_{chi_PC=}_{d=}.txt", c_arr)
# %%

#%%
L = wall_thickness
J_pore = get_pore_2(J_arr[:,:,0], L)
rho_pore = get_pore_2(ρ_arr[:, :]+1/2, L)
fig, ax = plt.subplots()
ax.plot(np.sum(J_pore*rho_pore, axis = 1), label = "$J_z$")
ax.axvline(x = L, color = "red")
ax.axvline(x = wall_thickness+L, color = "red")
ax.legend()

ax.axhline(y = J_z_total, color = "black")
ax.text(x = 0, y =J_z_total, s= "$J_{z}^{tot}$", ha = "left", va = "bottom", color = "black")
#ax.set_ylim(0 , J_z_total*1.1)

c_inflow = np.mean(get_pore(c_arr)[0])
c_outflow= np.mean(get_pore(c_arr)[1])
ax.text(x = wall_thickness, y=1, s= f"{c_inflow:.3f}", ha = "right", color = "red")
ax.text(x = wall_thickness*2, y=1, s= f"{c_outflow:.3f}", ha = "right", color = "red")

ax.set_xlabel("z")
ax.set_ylabel("$J_{z}$")

fig.set_size_inches(6,4)
if empty_pore:
    fig.savefig(f"fig/permeability/J_z_total_in_pore_sized_empty.pdf")
else:
    fig.savefig(f"fig/permeability/J_z_total_in_pore_sized_{chi=}_{chi_PC=}_{d=}.pdf")
# %%
fig, ax  = plt.subplots()
ax.set_aspect('equal')
levels = np.concatenate((np.arange(0, 0.05, 0.01), np.arange(0.05, 0.95, 0.05), np.arange(0.96, 1.0, 0.01)))
c = ax.contour(c_arr.T, origin = "lower", colors = "black", levels = levels)
ax.clabel(c, inline=True, zorder=2)

ax.add_patch(patches.Rectangle((l1-1, r-h-1), wall_thickness+2, h+2, color = "white", zorder=3))
ax.add_patch(patches.Rectangle((l1, r-h), wall_thickness, h, color = "grey", zorder=3))
ax.text(z/2, r/5*3, "WALL", ha="center", fontsize = 14)

ax.set_ylim(0,r)
ax.set_xlim(0,z)
ax.set_xlabel("$z$")
ax.set_ylabel("$r$")
ax.set_title("$c$")

if empty_pore:
    fig.savefig(f"fig/permeability/colloid_concentration_empty.pdf")
else:
    fig.savefig(f"fig/permeability/colloid_concentration_{chi=}_{chi_PC=}_{d=}.pdf")


# %%
fig, ax  = plt.subplots()
#c_arr_empty = np.loadtxt("c_arr_empty_pore.txt")
ax2 = ax.twinx()
ax2.plot(J_arr[:,0, 0]/2, label = "$j_z(r=0)$", color = "black")
J_z_mean = np.sum(J_arr[:,:,0]*(ρ_arr+1/2), axis = 1)/r**2*2
ax2.plot(J_z_mean, color = "grey", label = r"$\left<j_z\right>$")
ax.plot(c_arr[:,0], label = "$c(r=0)$")
ax.set_yscale("log")
#ax.plot(c_arr_empty[:,0])
ax.axvline(x = l1, color = "red")
ax.axvline(x = l1+wall_thickness, color = "red")
ax.legend(loc= 2)
ax2.legend(loc= 1)

ax.set_ylabel("$c/c_0$")
ax2.set_ylabel("$j_z$")
ax.set_xlabel("$z$")


#c_inflow = np.mean(get_pore(c_arr)[0])
#c_outflow= np.mean(get_pore(c_arr)[1])
#ax.text(x = l1, y=0, s= f"{c_inflow:.3f}", ha = "right", color = "red")
#ax.text(x = l1+wall_thickness, y=0, s= f"{c_outflow:.3f}", ha = "right", color = "red")

fig.set_size_inches(6,4)

if empty_pore:
    fig.savefig(f"fig/permeability/j_z_r=0_empty.pdf")
else:
    fig.savefig(f"fig/permeability/j_z_r=0_{chi=}_{chi_PC=}_{d=}.pdf")
#%%


# %%
fig, ax  = plt.subplots()
#c_arr_empty = np.loadtxt("c_arr_empty_pore.txt")
ax2 = ax.twinx()
ax2.plot(J_arr[int(l1+wall_thickness/2), :, 0], label = "$J_z(z=0)$", color = "black")
ax.plot(c_arr[int(l1+wall_thickness/2), :], label = "$c(z=0)$")
#ax.plot(c_arr_empty[:,0])
ax.legend(loc= 3)
ax2.legend(loc= 1)
ax.axvline(x = pore_radius, color = "red")
#ax.set_ylim(0)
ax.set_xlim(0, pore_radius*1.5)

fig.set_size_inches(6,4)

if empty_pore:
    fig.savefig(f"fig/permeability/j_z_z=0_empty.pdf")
else:
    fig.savefig(f"fig/permeability/j_z_z=0_{chi=}_{chi_PC=}_{d=}.pdf")
# %%
