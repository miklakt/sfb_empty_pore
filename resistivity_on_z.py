#%%
from calculate_fields_in_pore import *
import numpy as np

def make_analytical_resistivity_on_z(pore_radius, wall_thickness, d):
    pore_radius_ = pore_radius - d/2
    def func(z:float)->float:
        if abs(z)<=wall_thickness/2:
            return 1/(np.pi*pore_radius_**2)
        z_ = abs(z) - wall_thickness/2
        #return 1/(2*(pore_radius_+z_)*(pore_radius_+3*z_)*np.log(3))
        return 1/(4*z_**2 + pore_radius_**2)/np.pi
    return func

from scipy.ndimage import gaussian_filter1d

def smooth_peaks(arr, sigma):
    original_sum = np.sum(arr)
    smoothed = np.copy(arr)
    
    # Apply Gaussian filter excluding the edges
    smoothed[1:-1] = gaussian_filter1d(arr, sigma=sigma)[1:-1]

    # Adjust the smoothed values to maintain the integral
    smoothed_sum = np.sum(smoothed[1:-1])
    adjustment_factor = (original_sum - arr[0] - arr[-1]) / smoothed_sum
    smoothed[1:-1] *= adjustment_factor

    return smoothed
            
#%%
d=8
chi_PC = -1.5
chi_PS = 0.5
parameters = dict(
    a0 = 0.70585835,
    a1 = -0.31406453,
    chi_PC = chi_PC,
    chi = chi_PS,
    wall_thickness=52,
    pore_radius=26,
    d = d,
    sigma = 0.02,
    exclude_volume = True, 
    truncate_pressure = False,
    #method = "no_free_energy",
    method = "convolve",
    convolve_mode = "same",
    mobility_correction = "vol_average",
    mobility_model = "Rubinstein",
    mobility_model_kwargs = dict(prefactor = 1),
    # mobility_model = "none",
    # mobility_model_kwargs = {},
    #mobility_model = "Hoyst",
    #mobility_model_kwargs = {'alpha': 1.63, 'delta': 0.89, 'N':300},
    #mobility_model_kwargs = dict(beta = 8, nu = 0.76)
)

fields = {}
fields[chi_PC] = calculate_fields(**parameters)
chi_PC = -1.25
parameters["chi_PC"] = chi_PC
fields[chi_PC] = calculate_fields(**parameters)
chi_PC = -1.0
parameters["chi_PC"] = chi_PC
fields[chi_PC] = calculate_fields(**parameters)
# chi_PC = -0.75
# parameters["chi_PC"] = chi_PC
# fields[chi_PC] = calculate_fields(**parameters)
# chi_PC = -0.5
# parameters["chi_PC"] = chi_PC
# fields[chi_PC] = calculate_fields(**parameters)
# chi_PC = 0.0
# parameters["chi_PC"] = chi_PC
# fields[chi_PC] = calculate_fields(**parameters)
# parameters["chi_PS"] = parameters["chi"]
# del parameters["chi"]
# perm = calculate_permeability(**parameters,         
#             integration= "cylindrical_caps",
#             integration_kwargs = dict(spheroid_correction = True)
#             )
# %%
l1 = fields[chi_PC]["l1"]
wall_thickness = fields[chi_PC]["s"]
pore_radius = fields[chi_PC]["r"]
xlayers = fields[chi_PC]["xlayers"]
ylayers = fields[chi_PC]["ylayers"]
#field = fields_["conductivity"]
conduct = {
    chi_PC_:integrate_with_cylindrical_caps(field["conductivity"], l1, wall_thickness, pore_radius, xlayers, ylayers, spheroid_correction=True)
    for chi_PC_, field in fields.items()
    }
analytical_func = make_analytical_resistivity_on_z(pore_radius=26, wall_thickness=52, d=d)
# %%
#einstein_factor = 1/(3*np.pi*d)
fig, ax = plt.subplots()
x = np.arange(ylayers) - ylayers/2
analytical_resistivity = np.array([analytical_func(x_) for x_ in x])
#plt.plot(conduct[:int(292/2-26)])
norm = np.pi*pore_radius**2
[ax.plot(
    x, 
    smooth_peaks(conduct_**-1, 1.5)*norm,
    #conduct_**-1, 
    #label = f"$\chi_{{PC}} = {chi_PC_}$"
    label = f"${chi_PC_}$"
    ) 
    for chi_PC_, conduct_ in conduct.items() #if not chi_PC_==-0.5
    ]
ax.plot(
    x, 
    analytical_resistivity*norm, 
    #label="analytical", 
    color = "black"
    )
#ax.plot(x, perm["permeability_z"]**(-1)*einstein_factor)
ax.axvline(-wall_thickness/2, color = "black", linestyle = "--")
ax.axvline(wall_thickness/2, color = "black", linestyle = "--")
ax.set_xlabel("$z$")
ax.set_ylabel(r"$\rho D_0 \pi r_{pore}^2$")
#ax.set_yscale("log")
# ax.text(

#     0, analytical_resitivity[int(ylayers/2)], 
#     r"$\frac{1}{\pi r_{pore}^2}$", 
#     #transform = ax.transAxes, 
#     va = "bottom",
#     ha = "center",
#     fontsize = 14)
# ax.text(

#     0, 0.95, 
#     r"$\frac{1}{2 \ln3 (r_{pore} + z) (r_{pore} + 3z)}$", 
#     #transform = ax.transAxes, 
#     va = "top",
#     ha = "left",
#     fontsize = 14,
#     transform = ax.transAxes)
ax.legend(title = "$\chi_{PC}$")
#ax.set_ylim(0, 3.5/(np.pi*(pore_radius-d/2)**2))
ax.set_xlim(-100 ,100)
ax.set_ylim(0,5)
fig.set_size_inches(3,3)
fig.savefig(f"fig/resistivity_z_on_chi_PC_{chi_PS=}.svg")
#%%
d=8
chi_PC = -1.0
chi_PS = 0.3
parameters = dict(
    a0 = 0.70585835,
    a1 = -0.31406453,
    chi_PC = chi_PC,
    chi = chi_PS,
    wall_thickness=52,
    pore_radius=26,
    d = d,
    sigma = 0.02,
    exclude_volume = True, 
    truncate_pressure = False,
    #method = "no_free_energy",
    method = "convolve",
    convolve_mode = "same",
    mobility_correction = "vol_average",
    mobility_model = "Rubinstein",
    mobility_model_kwargs = dict(prefactor = 1),
    #mobility_model = "none",
    #mobility_model_kwargs = {},
    #mobility_model = "Hoyst",
    #mobility_model_kwargs = {'alpha': 1.63, 'delta': 0.89, 'N':300},
    #mobility_model_kwargs = dict(beta = 8, nu = 0.76)
)

fields = {}
fields[chi_PS] = calculate_fields(**parameters)
chi_PS = 0.5
parameters["chi"] = chi_PS
fields[chi_PS] = calculate_fields(**parameters)
chi_PS = 0.7
parameters["chi"] = chi_PS
fields[chi_PS] = calculate_fields(**parameters)
# parameters["chi_PS"] = parameters["chi"]
# del parameters["chi"]
# perm = calculate_permeability(**parameters,         
#             integration= "cylindrical_caps",
#             integration_kwargs = dict(spheroid_correction = True)
#             )
# %%
l1 = fields[chi_PS]["l1"]
wall_thickness = fields[chi_PS]["s"]
pore_radius = fields[chi_PS]["r"]
xlayers = fields[chi_PS]["xlayers"]
ylayers = fields[chi_PS]["ylayers"]
#field = fields_["conductivity"]
conduct = {
    chi_PS_:integrate_with_cylindrical_caps(field["conductivity"], l1, wall_thickness, pore_radius, xlayers, ylayers, spheroid_correction=True)
    for chi_PS_, field in fields.items()
    }
analytical_func = make_analytical_resistivity_on_z(pore_radius=26, wall_thickness=52, d=d)
# %%
#einstein_factor = 1/(3*np.pi*d)
fig, ax = plt.subplots()
x = np.arange(ylayers) - ylayers/2
analytical_resistivity = np.array([analytical_func(x_) for x_ in x])
norm = np.pi*pore_radius**2
[ax.plot(
    x, 
    smooth_peaks(conduct_**-1, 1.5)*norm,
    #label = f"$\chi_{{PS}} = {chi_PS_}$"
    label = f"${chi_PS_}$"
    ) 
    for chi_PS_, conduct_ in conduct.items()]
ax.plot(
    x, 
    analytical_resistivity*norm, 
    #label="analytical", 
    color = "black"
    )


ax.axvline(-wall_thickness/2, color = "black", linestyle = "--")
ax.axvline(wall_thickness/2, color = "black", linestyle = "--")
ax.axvline(-wall_thickness/2, color = "black", linestyle = "--")
ax.axvline(wall_thickness/2, color = "black", linestyle = "--")

ax.set_xlabel("$z$")
ax.set_ylabel(r"$\rho D_0 \pi r_{pore}^2$")
#ax.set_yscale("log")
# ax.text(
#     0, analytical_resitivity[int(ylayers/2)], 
#     r"$\frac{1}{\pi r_{pore}^2}$", 
#     #transform = ax.transAxes, 
#     va = "bottom",
#     ha = "center",
#     fontsize = 14)
# ax.text(
#     0, 0.95, 
#     r"$\frac{1}{2 \ln3 (r_{pore} + z) (r_{pore} + 3z)}$", 
#     #transform = ax.transAxes, 
#     va = "top",
#     ha = "left",
#     fontsize = 14,
#     transform = ax.transAxes)
ax.legend(title = "$\chi_{PS}$")
fig.set_size_inches(3,3)
ax.set_xlim(-100 ,100)
ax.set_ylim(0,5)
fig.savefig(f"fig/resistivity_z_on_chi_PS_{chi_PC=}.svg")
# %%
