import os, sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
os.chdir("..")
import drift_diffusion_stencil as drift_diffusion_stencil
from drift_diffusion_stencil import DriftDiffusionKernelFactory
from calculate_fields_in_pore import *
from heatmap_explorer import plot_heatmap_and_profiles
import tqdm
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as xp
except:
    import numpy as xp
#import numpy as xp

def pad_fields(fields, pad_sides, pad_top):
    padded_fields = {}
    padded_fields["xlayers"]=fields["xlayers"]+pad_sides*2
    padded_fields["ylayers"]=fields["ylayers"]+pad_top

    padded_fields["walls"]=xp.array(np.pad(
        fields["walls"],
        ((pad_sides, pad_sides), (0, pad_top)), 
        "edge",
        ))
    
    padded_fields["mobility"]=xp.array(np.pad(
        fields["mobility"],
        ((pad_sides, pad_sides), (0, pad_top)), 
        "constant", constant_values=(True, True)
        ))
    
    padded_fields["mobility"][padded_fields["walls"]==True]=0.0
    
    padded_fields["free_energy"]=xp.array(np.pad(
        fields["free_energy"],
        ((pad_sides, pad_sides), (0, pad_top)), 
        "constant", constant_values=(0.0, 0.0)
        ))
    return padded_fields
#%%
a0, a1 = [0.70585835, -0.31406453]
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness
d=6
chi_PC = -2.0
chi = 0.1
sigma = 0.02


fields_ = calculate_fields(
    a0, a1, d=d,
    chi_PC=chi_PC, chi=chi,
    sigma = sigma,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    exclude_volume=True,
    truncate_pressure=False,
    method= "convolve", 
    mobility_correction= "vol_average",
    mobility_model = "Rubinstein",
    mobility_model_kwargs = {"prefactor":1.0}
    )
#%%
fields = pad_fields(fields_, pad_sides = 100, pad_top = 200)

W_arr = fields["walls"]
D_arr = fields["mobility"]
#D_arr = xp.ones_like(W_arr, dtype = float)
U_arr = xp.zeros_like(W_arr)

zlayers = xp.shape(W_arr)[0]
rlayers = xp.shape(W_arr)[1]
differencing = "power_law"
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

output_filename = f"numerical_simulation/simulation_data/no_free_energy_{zlayers}_{rlayers}_{pore_radius}_{wall_thickness}_{d}_{chi}.txt"
#output_filename = f"numerical_simulation/simulation_data/empty_{zlayers}_{rlayers}_{pore_radius}_{wall_thickness}_{d}.txt"
try:
    c0 = xp.loadtxt(output_filename)
    drift_diffusion.c_arr = xp.array(c0)
    print("Previous calculation found")
except FileNotFoundError:
    print("No previous calculation found")
#%%
dt = 0.1
drift_diffusion.run_until(
    inflow_boundary, dt=dt,
    target_divJ_tot=1e-8,
    jump_every=None,
    timeout=1000
    )
#%%
c_arr = drift_diffusion.c_arr.get()
#%%
np.savetxt(output_filename, c_arr)
# %%
plot_heatmap_and_profiles(drift_diffusion.c_arr.get(), mask = drift_diffusion.W_arr.get(), contour = True)
plot_heatmap_and_profiles(drift_diffusion.div_J_arr.get(), mask = drift_diffusion.W_arr.get())
# %%
plt.plot(drift_diffusion.J_z_tot().get())
# %%
