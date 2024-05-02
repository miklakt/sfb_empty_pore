#%%
import os, sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
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

from scipy import ndimage
#import numpy as xp

def define_box_potential_pore(
    rlayers, 
    zlayers, 
    wall_thickness, 
    pore_radius, 
    potential, 
    l_r = 0,
    d = None,
    ):
    W_arr = xp.zeros((zlayers, rlayers))
    D_arr = xp.ones_like(W_arr)
    U_arr = xp.zeros_like(W_arr)

    l1 = int((zlayers - wall_thickness)/2)
    W_arr[l1:l1+wall_thickness+1, pore_radius:] = True
    if d is not None:
        W_arr = xp.array(ndimage.binary_dilation(W_arr.get(), structure = generate_circle_kernel(d)))

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
potential = 0
l_r = 0
d = 14

W_arr, D_arr, U_arr =  define_box_potential_pore(
                        rlayers, zlayers, 
                        wall_thickness, pore_radius,
                        potential,
                        l_r,
                        d
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
dt = 0.01
drift_diffusion.create_kernel(dt = dt)
def inflow_boundary(dd):
    dd.c_arr[0,:]=1 #source left
    dd.c_arr[:,-1]=dd.c_arr[:,-2] #mirror top
    dd.c_arr[-1,:]=0 #sink  right
#%%
drift_diffusion.run_until(
    inflow_boundary, dt=0.1,
    target_divJ_tot=1e-12,
    jump_every=None,
    timeout=600
    )
c_arr = drift_diffusion.c_arr.get()
#%%
np.savetxt(f"numerical_simulation/simulation_data/empty_{zlayers}_{rlayers}_{pore_radius}_{wall_thickness}_{d}.txt", c_arr)
#%%
plot_heatmap_and_profiles(drift_diffusion.c_arr.get(), mask = drift_diffusion.W_arr.get(), x0 = 250, y0=0, contour=True)
#%%
plot_heatmap_and_profiles(drift_diffusion.div_J_arr.get(), mask = drift_diffusion.W_arr.get(), x0 = 250, y0=0)
#%%
plot_heatmap_and_profiles(np.linalg.norm(drift_diffusion.J_arr, axis = 2).get(), contour = True)
#%%
plt.plot(drift_diffusion.J_z_tot().get())

# %%
drift_diffusion.c_arr = xp.array(c_arr)
drift_diffusion.c_arr[drift_diffusion.W_arr == True] = 0.0