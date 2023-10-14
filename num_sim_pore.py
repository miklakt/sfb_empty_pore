#%%
from drift_diffusion_stencil import DriftDiffusionKernelFactory
from calculate_fields_in_pore import *
from heatmap_explorer import plot_heatmap_and_profiles
import tqdm
import pickle
import h5py

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
method = dict(
    exclude_volume = True, 
    truncate_pressure = False, 
    method = "convolve_same", 
    mobility_correction = "vol_average"
)

a0, a1 = [0.70585835, -0.31406453]
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness
d = 16
chi_PC = -1.5
chi = 0.4
walls_only = False

fields_ = calculate_fields(
    a0, a1, d=d,
    chi_PC=chi_PC, chi=chi,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
    )

#%%
fields = pad_fields(fields_, pad_sides = 100, pad_top = 200)
#%%
W_arr = fields["walls"]

if walls_only:
    D_arr = xp.ones_like(W_arr)
    U_arr = xp.zeros_like(W_arr)
else:
    D_arr = fields["mobility"]
    U_arr = fields["free_energy"]

zlayers = xp.shape(W_arr)[0]
rlayers = xp.shape(W_arr)[1]
differencing = "power_law"
drift_diffusion = DriftDiffusionKernelFactory(
    W_arr=W_arr, D_arr = D_arr, U_arr = U_arr,
    differencing=differencing
    )

#%%
dt = 0.1
drift_diffusion.create_kernel(dt = dt)
def inflow_boundary(dd):
    dd.c_arr[0,:]=1 #source left
    dd.c_arr[:,-1]=dd.c_arr[:,-2] #mirror top
    #c_arr[:,0]=c_arr[:,1] #mirror bottom
    #dd.c_arr[:,-1]=dd.c_arr[:,-2] + xp.less_equal(dd.c_arr[:,-2] - dd.c_arr[:,-3], 0) # constant deriv 
    dd.c_arr[-1,:]=0 #sink  right

#%%
from drift_diffusion_stencil import SimulationManager
if walls_only:
     simulation_name = \
     f"simulation_data/{d=}_{zlayers=}_{rlayers=}_{dt=}_{differencing}.h5"
else:
     simulation_name = \
     f"simulation_data/{d=}_{chi=}_{chi_PC=}_{zlayers=}_{rlayers=}_{dt=}_{differencing}.h5"
s = SimulationManager(drift_diffusion, inflow_boundary, simulation_name)
#%%
drift_diffusion.c_arr = np.exp(-drift_diffusion.U_arr)
#%%
s.run(100, 1000)
s.run(100, 10000)
#%%
s.run(100, 100000)
# %%
plot_heatmap_and_profiles(drift_diffusion.c_arr.get(), mask = drift_diffusion.W_arr.get())
# %%
plt.plot(drift_diffusion.J_z_tot().get())
# %%
# %%
print(
    np.mean(drift_diffusion.J_z_tot().get()), 
    np.std(drift_diffusion.J_z_tot().get())
    )
# %%
