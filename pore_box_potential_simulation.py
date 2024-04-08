#%%
from drift_diffusion_stencil import SimulationManager#, SteadySimulationManager
from drift_diffusion_stencil import DriftDiffusionKernelFactory#, SteadyDriftDiffusionKernelFactory
from calculate_fields_in_pore import *
from heatmap_explorer import plot_heatmap_and_profiles
import tqdm
import pickle
import h5py

import matplotlib.pyplot as plt

try:
    import cupy as xp
except:
    import numpy as xp
#import numpy as xp

def define_box_potential_pore(rlayers, zlayers, wall_thickness, pore_radius, potential, l_r = 0):
    W_arr = xp.zeros((zlayers, rlayers))
    D_arr = xp.ones_like(W_arr)
    U_arr = xp.zeros_like(W_arr)

    l1 = int((zlayers - wall_thickness)/2)
    W_arr[l1:l1+wall_thickness+1, pore_radius:] = True
    U_arr[int(l1-l_r):int(l1+wall_thickness+l_r+1), :pore_radius] = potential

    D_arr[W_arr==True] = 0.0

    #return {"W_arr" : W_arr, "D_arr" : D_arr, "U_arr" : U_arr}
    return W_arr, D_arr, U_arr

def define_step_potential(rlayers, zlayers, potential):
    W_arr = xp.zeros((zlayers, rlayers))
    D_arr = xp.ones_like(W_arr)
    U_arr = xp.zeros_like(W_arr)
    U_arr[ int(zlayers/2):,: ] = potential
    return W_arr, D_arr, U_arr

#%%
rlayers = 400
zlayers = 600
pore_radius = 26
wall_thickness = 52
potential = 20
l_r = 20

W_arr, D_arr, U_arr =  define_box_potential_pore(
                        rlayers, zlayers, 
                        wall_thickness, pore_radius,
                        potential,
                        l_r
                        )

# W_arr, D_arr, U_arr =  define_step_potential(
#                         rlayers, zlayers, 
#                         potential)
differencing = "power_law"
#%%
drift_diffusion = DriftDiffusionKernelFactory(
    W_arr=xp.array(W_arr, dtype="int8"),
    D_arr = xp.array(D_arr),
    U_arr = xp.array(U_arr),
    differencing=differencing
    )

#%%
dt = 0.02
drift_diffusion.create_kernel(dt = dt)
def inflow_boundary(dd):
    dd.c_arr[0,:]=1 #source left
    dd.c_arr[:,-1]=dd.c_arr[:,-2] #mirror top
    #dd.c_arr[:,-1]=0
    #dd.c_arr[:,0]=dd.c_arr[:,1] #mirror bottom
    #dd.c_arr[:,-1]=dd.c_arr[:,-2] + xp.less_equal(dd.c_arr[:,-2] - dd.c_arr[:,-3], 0) # constant deriv 
    #dd.c_arr[:,-1]=2*dd.c_arr[:,-2] - dd.c_arr[:,-3]
    dd.c_arr[-1,:]=0 #sink  right
    #dd.c_arr[-1,:]=dd.c_arr[-2,:] #mirror  right
#%%
if (l_r is None) or (l_r == 0):
    simulation_name = f"simulation_data/box_potential_{zlayers}_{rlayers}_{potential}_{pore_radius}_{wall_thickness}_{dt=}_{differencing}.h5"
else:
    simulation_name = f"simulation_data/box_potential_{zlayers}_{rlayers}_{potential}_{pore_radius}_{wall_thickness}_{l_r=}_{dt=}_{differencing}.h5"
#simulation_name = f"simulation_data2/box_potential_{zlayers}_{rlayers}_{potential}_{pore_radius}_{wall_thickness}_{dt=}.h5"
s = SimulationManager(drift_diffusion, inflow_boundary, simulation_name)
#%%
# l1 = int((zlayers - wall_thickness)/2)
# drift_diffusion.c_arr[l1:l1+wall_thickness+1, :pore_radius+1] = 1000
#drift_diffusion.c_arr[int(zlayers/2):,:] = 1000
#%%
s.run(10,100)
#%%
s.run(10, 1000)
#%%
s.run(10, 10000)
#%%
s.run(10, 100000)
#%%
s.run(10, 1000000)
#%%
plot_heatmap_and_profiles(drift_diffusion.U_arr.get(), mask = drift_diffusion.W_arr.get(), x0 = 250, y0=0, contour=True)
#%%
plot_heatmap_and_profiles(drift_diffusion.J_arr[ :, :, 0].get(), mask = drift_diffusion.W_arr.get(), x0 = 250, y0=0)
#%%
plot_heatmap_and_profiles(np.linalg.norm(drift_diffusion.J_arr, axis = 2).get(), contour = True)
#%%
plt.plot(drift_diffusion.J_z_tot().get())
# %%
# %%
def get_flux_empty_pore_theory(D, r, L, c):
    return 2*D*r*c/(np.pi + 2*L/r)
# %%
print("potential",   "J_tot",    "J_tot_err")
print(
    potential,
    np.round(np.mean(drift_diffusion.J_z_tot().get()), 3), 
    np.round(np.std(drift_diffusion.J_z_tot().get()), 3),
    sep = ", "
    )
# %%
results = [
    (20, 0.574, 0.001),
    (10, 0.999, 0.001),
    (5, 1.265, 0.002),
    (3, 1.474, 0.002),
    (2, 2.034, 0.02),
    (1.5, 2.648, 0.019),
    (1, 3.629, 0.005),
    (0.5, 5.074, 0.008),
    (0, 6.877, 0.005),
    (-0.5, 8.887, 0.011),
    (-1.0, 10.69, 0.014),
    (-1.5, 12.09, 0.015),
    (-2.0, 13.045, 0.017),
    (-3.0, 13.986, 0.018),
    (-5.0, 14.43, 0.018),
    (-10.0, 14.967, 0.019),
    (-20.0, 15.769, 0.02),
    (-40.0, 16.313, 0.044),
]
results20=[
    (-40, 27.708, 0.047),
    (-20, 25.687, 0.033),
    (-10, 22.994, 0.03),
    (-5, 21.48, 0.028),
    (-2, 17.925, 0.023),
    (-1, 12.81, 0.016),
    (0, 6.877, 0.005),
    (1, 3.208, 0.004),
    (2, 1.7, 0.002),
    (5, 1.042, 0.001),
    (10, 0.818, 0.001),
    (20, 0.466, 0.001),
]

x = [r[0] for r in results]
j = [r[1] for r in results]
j_err = [r[2] for r in results]

x20 = [r[0] for r in results20]
j20 = [r[1] for r in results20]
j_err20 = [r[2] for r in results20]

flux_thick = get_flux_empty_pore_theory(1, 26,52, 1)
flux_thin = get_flux_empty_pore_theory(1, 26, 0, 1)

fig, ax = plt.subplots()

ax.axhline(flux_thick, color = "black", linestyle = "--", label = "empty")
ax.axhline(flux_thin, color = "black", linestyle = ":", label = "empty, thin")



ax.plot(x, j, label = "$l_r = 0$", linewidth = 0.1, marker = "o")
ax.plot(x20, j20, label = "$l_r = 20$", linewidth = 0.1, marker = "o")

ax.set_xlabel("$\Delta F / k_BT$")
ax.set_ylabel("J")

ax.legend()

#ax.set_yscale("log")

# %%
