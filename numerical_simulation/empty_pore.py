#%%
import os, sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
import drift_diffusion_stencil as drift_diffusion_stencil
from drift_diffusion_stencil import DriftDiffusionKernelFactory
from calculate_fields_in_pore import *
from heatmap_explorer import plot_heatmap_and_profiles
import tqdm
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt

__cupy__ = True
if __cupy__:
    import cupy as xp
else:
    import numpy as xp
#import numpy as xp
#%%
empty = True
a0, a1 = [0.70585835, -0.31406453]
pore_radius = 25 # pore radius
wall_thickness = 52 # wall thickness
d=20
rlayers = 300
zlayers = 600

l1 = int((zlayers-wall_thickness)/2)

os.chdir("..")

W_arr = np.zeros((zlayers,rlayers))
W_arr[l1:l1+wall_thickness+1, pore_radius:] = True
#%%
W_arr = xp.array(W_arr, dtype = "int8")
D_arr = xp.ones_like(W_arr, dtype = float)
U_arr = xp.zeros_like(W_arr)

differencing = "power_law"
#%%
drift_diffusion = DriftDiffusionKernelFactory(
    W_arr=W_arr, D_arr = D_arr, U_arr = U_arr,
    differencing=differencing
    )

#%%
def inflow_boundary(dd):
    dd.c_arr[0,:]=1 #source left
    dd.c_arr[:,-1]=dd.c_arr[:,-2] #mirror top
    #c_arr[:,0]=c_arr[:,1] #mirror bottom
    #dd.c_arr[:,-1]=dd.c_arr[:,-2] + xp.less_equal(dd.c_arr[:,-2] - dd.c_arr[:,-3], 0) # constant deriv 
    dd.c_arr[-1,:]=0 #sink  right

output_filename = here + f"/simulation_data/empty_{zlayers}_{rlayers}_{pore_radius}_{wall_thickness}_{d}.txt"
try:
    c0 = xp.loadtxt(output_filename)
    drift_diffusion.c_arr = xp.array(c0)
    print("Previous calculation found")
except FileNotFoundError:
    print("No previous calculation found")
#%%
dt = 0.2
drift_diffusion.run_until(
    inflow_boundary, dt=dt,
    target_divJ_tot=1e-4,
    jump_every=None,
    timeout=1000
    )
#%%
c_arr = drift_diffusion.c_arr
#%%
#drift_diffusion.c_arr = xp.array(np.loadtxt("temp.txt"))
np.savetxt("temp.txt", c_arr)
np.savetxt(output_filename, c_arr)
# %%
plot_heatmap_and_profiles(drift_diffusion.c_arr.get(), mask = drift_diffusion.W_arr.get())
plot_heatmap_and_profiles(drift_diffusion.div_J_arr.get(), mask = drift_diffusion.W_arr.get())
# %%
plt.plot(drift_diffusion.J_z_tot().get())
# %%
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rc('hatch', color='darkgreen', linewidth=9)
fig, ax = plt.subplots()
c_arr_ =  xp.vstack((xp.flip(c_arr.T), c_arr.T[:,::-1])).get()
W_arr_ = xp.vstack((xp.flip(W_arr.T), W_arr.T[:,::-1])).get()
c_arr_ = np.ma.array(c_arr_, mask = W_arr_)

ylayers, xlayers = np.shape(c_arr_)

levels_ = np.arange(0.0, 1.0, 0.05)
# levels = np.sort(np.concatenate([[0.93],[0.07],
#                                 np.arange(0.95, 1.00, 0.01),
#                                  #np.arange(0.98, 1.00, 0.005),
#                                  np.arange(0.01, 0.06, 0.01),
#                                  #np.arange(0.005, .035, 0.005),
#                                  ]))
cs = ax.contour(
    c_arr_, 
    origin = "lower", 
    colors = "black", 
    levels = levels_,
    # linewidths = 1.5,
    extent = [-xlayers/2, xlayers/2, -ylayers/2, ylayers/2]
    )

# cs = ax.contour(
#     c_arr_, 
#     origin = "lower", 
#     colors = "black", 
#     levels = levels,
#     linewidths = 0.75,
#     extent = [-xlayers/2, xlayers/2, -ylayers/2, ylayers/2]
#     )

# cs = ax.contour(c_arr_, origin = "lower", colors = "black", levels = levels,

# extent = [-xlayers/2, xlayers/2, -ylayers/2, ylayers/2])
#ax.clabel(cs)

ax.axis('equal')
#s=52
#pore_radius=26
r_cut= ylayers/2
p = mpatches.Rectangle((-wall_thickness/2-1, -r_cut-1), wall_thickness+1, r_cut-pore_radius+1, hatch='/', facecolor = "green")
ax.add_patch(p)
p = mpatches.Rectangle((-wall_thickness/2-1, r_cut-1), wall_thickness+1, -r_cut+pore_radius+1, hatch='/', facecolor = "green")
ax.add_patch(p)

ax.set_xlim(-65, 65)
ax.set_ylim(-40, 40)

ax.set_xlabel("$z$")
ax.set_ylabel("$r$")
fig.set_size_inches(4,4)
fig.savefig("/home/ml/Studium/sfb_empty_pore/fig/empty_pore_contour2.svg")
# %%
