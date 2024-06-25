#%%
from calculate_fields_in_pore import *
import numpy as np

def make_analytical_resistivity_on_z(pore_radius, wall_thickness, d):
    pore_radius_ = pore_radius - d/2
    def func(z:float)->float:
        if abs(z)<=wall_thickness/2:
            return 1/(np.pi*pore_radius_**2)
        z_ = abs(z) - wall_thickness/2
        return 1/(2*(pore_radius_+z_)*(pore_radius_+3*z_)*np.log(3))
    return func
            
#%%
d=8
chi_PC = -1.5
parameters = dict(
    a0 = 0.70585835,
    a1 = -0.31406453,
    chi_PC = chi_PC,
    chi = 0.5,
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
fields[chi_PC] = calculate_fields(**parameters)
chi_PC = -1.0
parameters["chi_PC"] = chi_PC
fields[chi_PC] = calculate_fields(**parameters)
chi_PC = -0.5
parameters["chi_PC"] = chi_PC
fields[chi_PC] = calculate_fields(**parameters)
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
    chi_PC_:integrate_with_cylindrical_caps(field["conductivity"], l1, wall_thickness, pore_radius, xlayers, ylayers, spheroid_correction=False)
    for chi_PC_, field in fields.items()
    }
analytical_func = make_analytical_resistivity_on_z(pore_radius=26, wall_thickness=52, d=d)
# %%
#einstein_factor = 1/(3*np.pi*d)
fig, ax = plt.subplots()
x = np.arange(ylayers) - ylayers/2
analytical_resitivity = [analytical_func(x_) for x_ in x]
#plt.plot(conduct[:int(292/2-26)])
[ax.plot(x, conduct_**-1, label = f"$\chi_{{PC}} = {chi_PC_}$") for chi_PC_, conduct_ in conduct.items() if not chi_PC_==-0.5]
ax.plot(x, analytical_resitivity, label="analytical", color = "black")
#ax.plot(x, perm["permeability_z"]**(-1)*einstein_factor)
ax.axvline(-wall_thickness/2, color = "black", linestyle = "--")
ax.axvline(wall_thickness/2, color = "black", linestyle = "--")
ax.set_xlabel("$z$")
ax.set_ylabel(r"$\rho D_0$")
#ax.set_yscale("log")
ax.text(
    0, analytical_resitivity[int(ylayers/2)], 
    r"$\frac{1}{\pi r_{pore}^2}$", 
    #transform = ax.transAxes, 
    va = "bottom",
    ha = "center",
    fontsize = 14)
ax.text(
    0, 0.95, 
    r"$\frac{1}{2 \ln3 (r_{pore} + z) (r_{pore} + 3z)}$", 
    #transform = ax.transAxes, 
    va = "top",
    ha = "left",
    fontsize = 14,
    transform = ax.transAxes)
ax.legend()
fig.set_size_inches(5,3.5)
# %%
# fields["volume"] = np.ones_like(fields["phi"])
# fields["volume"][:] = 2*np.pi*(np.arange(fields["xlayers"])+1)
# einstein_factor = 1/(3*np.pi*d)
# R_thin = 1/empty_pore_permeability(1, pore_radius-d/2, 0)/einstein_factor/2
# R_pore = wall_thickness/(np.pi*(pore_radius-d/2)**2)/einstein_factor
# plt.legend()
# print(f"{R_thin=} {R_pore=}")
#np.sum(conduct[l1:l1+wall_thickness+1]**(-1))/einstein_factor
# %%
# r_cut = 50
# z_cut = 30
# plot_heatmap(fields, r_cut, z_cut, keys = ["corrected_phi", "free_energy", "osmotic", "conductivity"])
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
analytical_resitivity = [analytical_func(x_) for x_ in x]

[ax.plot(x, conduct_**-1, label = f"$\chi_{{PS}} = {chi_PS_}$") for chi_PS_, conduct_ in conduct.items()]
ax.plot(x, analytical_resitivity, label="analytical", color = "black")


ax.axvline(-wall_thickness/2, color = "black", linestyle = "--")
ax.axvline(wall_thickness/2, color = "black", linestyle = "--")
ax.axvline(-wall_thickness/2, color = "black", linestyle = "--")
ax.axvline(wall_thickness/2, color = "black", linestyle = "--")

ax.set_xlabel("$z$")
ax.set_ylabel(r"$\rho D_0$")
ax.set_yscale("log")
ax.text(
    0, analytical_resitivity[int(ylayers/2)], 
    r"$\frac{1}{\pi r_{pore}^2}$", 
    #transform = ax.transAxes, 
    va = "bottom",
    ha = "center",
    fontsize = 14)
ax.text(
    0, 0.95, 
    r"$\frac{1}{2 \ln3 (r_{pore} + z) (r_{pore} + 3z)}$", 
    #transform = ax.transAxes, 
    va = "top",
    ha = "left",
    fontsize = 14,
    transform = ax.transAxes)
ax.legend()
fig.set_size_inches(5,3.5)
# %%
