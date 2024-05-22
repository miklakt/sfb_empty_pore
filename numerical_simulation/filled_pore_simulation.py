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
#d_ = np.arange(6, 22, 2)
#d_ = [10]
d = 6
chi_PC = -2.0
chi = 0.1
sigma = 0.02

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
    mobility_model_kwargs = {"prefactor":1.0}
    )
#%%
fields = pad_fields(fields_, pad_sides = 100, pad_top = 200)
#fields = pad_fields(fields_, pad_sides = 0, pad_top = 0)
W_arr = fields["walls"]
D_arr = fields["mobility"]
U_arr = fields["free_energy"]

zlayers = xp.shape(W_arr)[0]
rlayers = xp.shape(W_arr)[1]
differencing = "power_law"
drift_diffusion = DriftDiffusionKernelFactory(
    W_arr=W_arr, D_arr = D_arr, U_arr = U_arr,
    differencing=differencing
    )

no_energy_calculation = f"numerical_simulation/simulation_data/no_free_energy_{zlayers}_{rlayers}_{pore_radius}_{wall_thickness}_{d}_{chi}.txt"
output_filename = f"numerical_simulation/simulation_data/filled_{zlayers}_{rlayers}_{pore_radius}_{wall_thickness}_{d}_{chi}_{chi_PC}.txt"

#%%
try:
    c0 = xp.loadtxt(output_filename)
    print("Previous calculation found")
except FileNotFoundError:
    print("No previous calculation found")
    try:
        c0 = xp.loadtxt(no_energy_calculation)
        c0 = c0*xp.exp(-drift_diffusion.U_arr)
        print("Initial guess is found")
    except FileNotFoundError:
        print("Initial guess is not found")

drift_diffusion.c_arr = c0
#%%
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
dt = 0.25
drift_diffusion.run_until(
    inflow_boundary, dt=dt,
    target_divJ_tot=1e-4,
    jump_every=10,
    timeout=6000,
    max_jump=1e30,
    sigmoid_steepness=1
    )
#%%
c_arr = drift_diffusion.c_arr.get()
np.savetxt("tmp.txt", c_arr)
#%%
np.savetxt(output_filename, c_arr)
# %%
plot_heatmap_and_profiles(drift_diffusion.c_arr.get(), mask = drift_diffusion.W_arr.get())
plot_heatmap_and_profiles(drift_diffusion.div_J_arr.get(), mask = drift_diffusion.W_arr.get())
# %%
plt.plot(drift_diffusion.J_z_tot().get())
# %%
print("d",   "chi_PS",    "chi_PC",   "J_tot",    "J_tot_err")
print(
    d,
    chi,
    chi_PC,
    np.round(np.mean(drift_diffusion.J_z_tot().get()), 4),
    np.round(np.std(drift_diffusion.J_z_tot().get()), 4),
    sep = ", "
    )
#%%
import pandas as pd
results = pd.DataFrame(columns=["d",   "chi_PS",    "chi_PC",   "J_tot",    "J_tot_err"],
    data = [
        (4, 0.5, -1.25, 7.7173, 0.0177),
        (10, 0.5, -1.25, 23.4871, 0.0568),
        (16, 0.5, -1.25, 37.8336, 0.247),
        (10, 0.3, -1.5, 16.0219, 0.0384),
        (12, 0.3, -1.5, 14.3025, 0.0358),
        (14, 0.3, -1.5, 7.9092, 0.0212),
        (16, 0.3, -1.5, 1.8923, 0.0057),
        (8, 0.3, -1.5, 14.0949, 0.033),
        (6, 0.3, -1.5, 10.9856, 0.0018),
        (4, 0.3, -1.5, 8.5315, 0.0012),
        (8, 0.1, -2.0, 35.146, 0.0812),
        (12, 0.1, -2.0, 54.8079, 0.1392),
        (16, 0.1, -2.0, 56.6125, 1.7755),
        (20, 0.1, -2.0, 1.84, 5),
        (18, 0.1, -2.0, 40.3837, 2.1726),
        (14, 0.1, -2.0, 58.1557, 0.5443),
        (4, 0.1, -2.0, 12.5677, 0.0287),
        (6, 0.1, -2.0, 21.755, 0.0028),
        (10, 0.3, -1.75, 46.8653, 0.1363),
        (4, 0.1, -1.75, 9.0709, 0.0208),
        (6, 0.1, -1.75, 11.6507, 0.0269),
        (8, 0.1, -1.75, 14.1156, 0.033),
        (10, 0.1, -1.75, 14.2061, 0.034),
        (12, 0.1, -1.75, 9.8666, 0.0245),
        (14, 0.1, -1.75, 3.3862, 0.0089),
        (16, 0.1, -1.75, 0.4119, 0.0011),
        (18, 0.1, -1.75, 0.0144, 0.0005),
        (4, 0.3, -1.75, 12.31, 0.0281),
        (6, 0.3, -1.75, 21.8729, 0.0501),
        (8, 0.3, -1.75, 35.5921, 0.082),
        (10, 0.3, -1.75, 46.8666, 0.1091),
        (12, 0.3, -1.75, 53.1488, 0.8342),
        (14, 0.3, -1.75, 56.2245, 0.3548),
        (20, 0.3, -1.75, 56.7624, 1.5648),
    ]
)
# %%
from scipy.optimize import minimize

objective = drift_diffusion.get_div_J_arr_on_c_arr(
    inflow_boundary, dt=0.001, smooth_over = 1, steps = 10
    )

bnds = np.zeros((zlayers*rlayers,2))
bnds[:,0] = 0.0
bnds[:,1] = None
#x0 = c0.flatten().get()
x0 = drift_diffusion.c_arr.flatten().get()
# %%
result = minimize(objective, x0, bounds = bnds, method="BFGS")
# %%
