#%%
import drift_diffusion_stencil as drift_diffusion_stencil
from drift_diffusion_stencil import DriftDiffusionKernelFactory
from calculate_fields_in_pore import *
from heatmap_explorer import plot_heatmap_and_profiles
import tqdm
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import csv
prev_cwd = os.getcwd()
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
os.chdir("..")


__cupy__ = True
if __cupy__:
    import cupy as xp
else:
    import numpy as xp

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
for d in np.arange(4, 22, 2):
    a0, a1 = [0.7, -0.3]
    pore_radius = 26 # pore radius
    wall_thickness = 52 # wall thickness
    #d_ = np.arange(6, 22, 2)
    #d_ = [10]
    # d = 30
    chi_PC = -1.3
    chi_PS = 0.5
    sigma = 0.02

    #for d in d_:
    fields_ = calculate_fields(
        a0=a0, a1=a1, d=d,
        chi_PC = chi_PC, chi_PS = chi_PS,
        sigma = sigma,
        wall_thickness=wall_thickness,
        pore_radius=pore_radius,
        mobility_model_kwargs = {"prefactor":30.0**(0.5)}
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

    no_energy_calculation = here + f"/simulation_data/no_free_energy_{zlayers}_{rlayers}_{pore_radius}_{wall_thickness}_{d}_{chi_PS}.txt"
    empty_pore_calculation = here + f"/simulation_data/empty_{zlayers}_{rlayers}_{pore_radius}_{wall_thickness}_{d}.txt"
    output_filename = here + f"/simulation_data/filled_{zlayers}_{rlayers}_{pore_radius}_{wall_thickness}_{d}_{chi_PS}_{chi_PC}.txt"

    #%%
    try:
        c0 = xp.loadtxt(output_filename)
        print("Previous calculation found")
    except FileNotFoundError:
        print("No previous calculation found")
        try:
            c0 = xp.loadtxt(no_energy_calculation)
            c0 = c0*xp.exp(-drift_diffusion.U_arr)#*xp.exp(drift_diffusion.D_arr-1)
            print("Initial guess is found")
        except FileNotFoundError:
            try:
                c0 = xp.loadtxt(empty_pore_calculation)
                c0 = c0*xp.exp(-drift_diffusion.U_arr)#*xp.exp(1-drift_diffusion.D_arr)
                print("Initial guess is empty pore")
            except FileNotFoundError:
                print("Initial guess is not found")
    drift_diffusion.c_arr = c0
    #%%
    x_center = int(xp.shape(drift_diffusion.c_arr)[0]/2)
    def inflow_boundary(dd):
        dd.c_arr[0,:]=1 #source left
        dd.c_arr[:,-1]=dd.c_arr[:,-2] #mirror top
        #c_arr[:,0]=c_arr[:,1] #mirror bottom
        #dd.c_arr[:x_center,-1]= dd.c_arr[:x_center,-2] + xp.greater_equal(dd.c_arr[:x_center,-2] - dd.c_arr[:x_center,-3],0) # constant deriv before the membrane
        #dd.c_arr[x_center:,-1]= dd.c_arr[x_center:,-2] + xp.less_equal(dd.c_arr[x_center:,-2] - dd.c_arr[x_center:,-3],0)
        dd.c_arr[-1,:]=0 #sink  right
    #%%
    # c_arr = np.loadtxt("tmp.txt")
    # drift_diffusion.c_arr = xp.array(c_arr)
    #%%
    dt = 0.1
    drift_diffusion.run_until(
        inflow_boundary, dt=dt,
        target_divJ_tot=1e-6,
        jump_every=10,
        timeout=12000,
        max_jump=1e-5,
        sigmoid_steepness=1,
        # jump_if_change = 1e-2
        )
    #%%
    c_arr = drift_diffusion.c_arr.get()
    # #np.savetxt("tmp.txt", c_arr)
    #%%
    np.savetxt(output_filename, c_arr)
    # %%
    # plot_heatmap_and_profiles(drift_diffusion.c_arr.get(), mask = drift_diffusion.W_arr.get())
    # #%%
    # plot_heatmap_and_profiles(drift_diffusion.div_J_arr.get(), mask = drift_diffusion.W_arr.get())
    # # %%
    # %matplotlib QtAgg
    # plot_heatmap_and_profiles(drift_diffusion.J_arr.get()[:,:,1], mask = drift_diffusion.W_arr.get())
    # #%%
    # plt.plot(drift_diffusion.J_z_tot().get())
    # %%
    print("d",   "chi_PS",    "chi_PC",   "J_tot",    "J_tot_err")
    print(
        d,
        chi_PS,
        chi_PC,
        np.round(np.mean(drift_diffusion.J_z_tot().get()), 4),
        np.round(np.std(drift_diffusion.J_z_tot().get()), 4),
        sep = ", ",
        )
    #%%
    with open("numeric_simulation_results_.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
        (d,
        chi_PS,
        chi_PC,
        np.round(np.mean(drift_diffusion.J_z_tot().get()), 4),
        np.round(np.std(drift_diffusion.J_z_tot().get()), 4),)
        )
    print("Tuple appended successfully!")
#%%
os.chdir(prev_cwd)
# %%
