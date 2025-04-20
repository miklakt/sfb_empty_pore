#%%
import os, sys
import drift_diffusion_stencil as drift_diffusion_stencil
from drift_diffusion_stencil import DriftDiffusionKernelFactory
#%%
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
os.chdir("..")
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

def get_half(fields):
    z_half = int(np.shape(fields["walls"])[0]/2)
    for key in ['walls', 'mobility', 'free_energy']:
        fields[key] = fields[key][:z_half, :]
#%%
a0, a1 = [0.70585835, -0.31406453]
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness
#d_ = np.arange(6, 22, 2)
#d_ = [10]
d = 12
chi_PC = -1.0
chi = 0.5
sigma = 0.02
prefactor = 30.0
#for d in d_:
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
    mobility_model_kwargs = {"prefactor":prefactor}
    )
#%%
fields = pad_fields(fields_, pad_sides = 100, pad_top = 200)
half = False
if half: get_half(fields)
#%%
W_arr = fields["walls"]
D_arr = fields["mobility"]
U_arr = fields["free_energy"]

empty = True
if empty:
    D_arr = xp.ones_like(W_arr, dtype = float)
    U_arr = xp.zeros_like(W_arr)


zlayers = xp.shape(W_arr)[0]
rlayers = xp.shape(W_arr)[1]
differencing = "power_law"
drift_diffusion = DriftDiffusionKernelFactory(
    W_arr=W_arr, D_arr = D_arr, U_arr = U_arr,
    differencing=differencing
    )

#%%
dt = 0.2
drift_diffusion.create_kernel(dt = dt)
def inflow_boundary(dd):
    dd.c_arr[0,:]=1 #source left
    dd.c_arr[:,-1]=dd.c_arr[:,-2] #mirror top
    #c_arr[:,0]=c_arr[:,1] #mirror bottom
    #dd.c_arr[:,-1]=dd.c_arr[:,-2] + xp.less_equal(dd.c_arr[:,-2] - dd.c_arr[:,-3], 0) # constant deriv 
    dd.c_arr[-1,:]=0 #sink  right

#%%
# c_arr = np.loadtxt("tmp.txt")
# drift_diffusion.c_arr = xp.array(c_arr)
#%%
# drift_diffusion.run_until(
#     inflow_boundary, dt=0.001, 
#     target_divJ_tot=1e-3, 
#     jump_every=None, 
#     max_jump=1.2, 
#     timeout = 30,
#     check_every=1000,
#     jump_if_change=1e-1,
#     sigmoid_steepness=100
#     )
# c_arr = drift_diffusion.c_arr.get()
# np.savetxt("tmp.txt", c_arr)
#%%
from drift_diffusion_stencil import SimulationManager
if half:
    simulation_name = \
    f"numerical_simulation/simulation_data/{d=}_{zlayers=}_{rlayers=}_{chi=}_{chi_PC=}_{dt=}_{differencing}_half.h5"
elif empty:
    simulation_name = \
    f"numerical_simulation/simulation_data/{d=}_{zlayers=}_{rlayers=}_{chi=}_{chi_PC=}_{dt=}_{differencing}_empty.h5"
else:
    simulation_name = \
    f"numerical_simulation/simulation_data/{d=}_{zlayers=}_{rlayers=}_{chi=}_{chi_PC=}_{dt=}_{differencing}.h5"
s = SimulationManager(drift_diffusion, inflow_boundary, simulation_name)
#%%
# c0 = xp.tile(xp.linspace(1,0, zlayers), (rlayers,1)).T
# c0 = c0*np.exp(-drift_diffusion.U_arr)/2
# drift_diffusion.c_arr = c0
#%%
s.run(1, 100)
#%%
s.run(10, 100)
#%%
s.run(10, 1000)
#%%
s.run(10, 10000)
#%%
s.run(10, 100000)
#%%
s.run(5, 1000000)
#%%
#s.run(10, 10000000)
# %%
plot_heatmap_and_profiles(drift_diffusion.c_arr.get(), mask = drift_diffusion.W_arr.get())
#plot_heatmap_and_profiles(drift_diffusion.div_J_arr.get(), mask = drift_diffusion.W_arr.get())
#%%
plot_heatmap_and_profiles(drift_diffusion.c_arr.get()*np.exp(drift_diffusion.U_arr.get()), mask = drift_diffusion.W_arr.get())
# %%
plt.plot(drift_diffusion.J_z_tot().get())
# %%
print("d",   "chi_PS",    "chi_PC",   "J_tot",    "J_tot_err")
print(
    d,
    chi,
    chi_PC,
    np.round(np.mean(drift_diffusion.J_z_tot().get()), 4),
    np.round(np.std(drift_diffusion.J_z_tot().get()), 6),
    sep = ", "
    )

# %%

# %%
c = drift_diffusion.c_arr[:,0].get()
output_filename = f"numerical_simulation/simulation_data/empty_{zlayers}_{rlayers}_{pore_radius}_{wall_thickness}_{d}.txt"
c0 = np.loadtxt(output_filename)[:,0]
fig, axs = plt.subplots(nrows = 2, sharex = True)

x = np.arange(0, zlayers) - zlayers/2

ax = axs[0]
ax.plot(x, c, label = "$c(z, r=0)$")
ax.plot(x, c0, label = "$c_{empty}(z, r=0)$")
ax.set_ylim(0)

ax = axs[1]
ax.plot(x, c, label = "$c(z, r=0)$")
ax.plot(x, c0, label = "$c_{empty}(z, r=0)$")
ax.set_ylim(0, 1.2)


ax.set_xlabel("$z$")
ax.set_ylabel("$c/c_0$")
ax.legend(loc = "lower left")
fig.set_size_inches(4,3)
#fig.savefig("fig/streamlines/total_flux_z.svg", transparent = True)
# %%
simulation_results = h5py.File(simulation_name, "r")
# %%
nsteps = len(simulation_results["timestep"])
J_tot_in = np.array([simulation_results["J_z_tot"][i][1] for i in range(nsteps)])
J_tot_out = np.array([simulation_results["J_z_tot"][i][-1] for i in range(nsteps)])


fig, ax = plt.subplots()
ax.plot(simulation_results["timestep"][1:], J_tot_in[1:], label = "$J_{in}$")
ax.plot(simulation_results["timestep"][1:], J_tot_out[1:], label = "$J_{out}$")
#ax.plot(simulation_results["timestep"][1:], J_tot_in[1:]/J_tot_out[1:], label = "$J_{in}/J_{out}$")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("$\int^{r_{pore}} j(r,z) dr$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-4,1e4)
# %%
