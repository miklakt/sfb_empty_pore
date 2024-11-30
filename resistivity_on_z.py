#%%
from calculate_fields_in_pore import *
import numpy as np
import matplotlib
matplotlib.rc('hatch', color='darkgreen', linewidth=9)

def make_analytical_resistivity_on_z(pore_radius, wall_thickness, func_type = "elliptic"):
    def func(z:float)->float:
        if abs(z)<=wall_thickness/2:
            return 1/(np.pi*pore_radius**2)
        z_ = abs(z) - wall_thickness/2
        #return 1/(2*(pore_radius_+z_)*(pore_radius_+3*z_)*np.log(3))
        if func_type == "quad":
            return 1/(4*z_**2 + pore_radius**2)/np.pi
        elif func_type == "elliptic":
            return 1/(2*z_**2 + 2*pore_radius**2)/np.pi
        else:
            raise ValueError("Wrong func type")
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
chi_PCs = [-1.2, -1.1, -1.0, -0.9]
# c chi_PCs = [-1.0]
chi_PS = 0.5
pore_radius = 26
wall_thickness = 52
Haberman_correction_ = True
parameters = dict(
    a0 = 0.70585835,
    a1 = -0.31406453,
    #chi_PC = chi_PC,
    chi = chi_PS,
    wall_thickness=wall_thickness,
    pore_radius=pore_radius,
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
    #mobility_model_kwargs = dict(beta = 8, nu = 0.76),
    Haberman_correction=Haberman_correction_
)

fields = {}
for chi_PC in chi_PCs:
    parameters.update(dict(chi_PC = chi_PC))
    fields[chi_PC] = calculate_fields(**parameters)
# %%
l1 = fields[chi_PC]["l1"]
wall_thickness = fields[chi_PC]["s"]
pore_radius = fields[chi_PC]["r"]
xlayers = fields[chi_PC]["xlayers"]
ylayers = fields[chi_PC]["ylayers"]
conduct = {
    chi_PC_:integrate_with_cylindrical_caps(field["conductivity"], 
        l1, wall_thickness, pore_radius,
        xlayers, ylayers,
        spheroid_correction=True
        )
    for chi_PC_, field in fields.items()
    }

empty_pore_arr = np.array(~fields[-1.0]["walls"], dtype="float")
if Haberman_correction_:
    wall_drag_correction = Haberman_correction_approximant(d, pore_radius)
    empty_pore_arr[l1:l1+wall_thickness, 0:pore_radius] = empty_pore_arr[l1:l1+wall_thickness, 0:pore_radius]/wall_drag_correction

#%%
conduct_empty = integrate_with_cylindrical_caps(
    empty_pore_arr,
    int(l1-d/2), int(wall_thickness+d), pore_radius,
    xlayers, ylayers,
    spheroid_correction=True)

analytical_func =\
    make_analytical_resistivity_on_z(
        pore_radius=pore_radius-d/2,
        wall_thickness=wall_thickness+d,
        )
# %%
#einstein_factor = 1/(3*np.pi*d)
fig, ax = plt.subplots()
x = np.arange(ylayers) - ylayers/2
x_ = np.linspace(0,ylayers, 1000) - ylayers/2
analytical_resistivity_elliptic = np.array([analytical_func(x__) for x__ in x_])#/einstein_factor


norm = np.pi*(pore_radius-d/2)**2
#norm =1
for chi_PC_, conduct_ in conduct.items():
    ax.plot(
        x, 
        smooth_peaks(conduct_**-1, 1)*norm,
        #conduct_**-1*norm, 
        label = f"$\chi_{{PC}} = {chi_PC_}$",
        #label = f"${chi_PC_}$"
    ) 


ax.plot(
    x, 
    #analytical_resistivity_elliptic*norm,
    conduct_empty**(-1)*norm,
    label="analytical", 
    color = "black"
    )


# ax.plot(
#     x_, 
#     analytical_resistivity_elliptic*norm,
#     #conduct_empty**(-1)*norm,
#     label="analytical2", 
#     color = "darkred"
#     )

ax.axhline(y = 1, color = "black", linewidth = 0.5)

trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
p = matplotlib.patches.Rectangle(
    (-wall_thickness/2, 0), 
    wall_thickness, 1, 
    hatch='/', 
    facecolor = "green", 
    transform = trans,
    #alpha = 0.1
    )
#ax.add_patch(p)

#ax.plot(x, perm["permeability_z"]**(-1)*einstein_factor)
ax.axvline(-wall_thickness/2, color = "black", linestyle = "--")
ax.axvline(wall_thickness/2, color = "black", linestyle = "--")
ax.set_xlabel("$z$")
ax.set_ylabel(r"$\rho D_0 \pi r_{pore}^2$")

#ax.legend(title = "$\chi_{PC}$")
ax.set_xlim(-100, 100)
ax.set_ylim(0,5.5)
fig.set_size_inches(2.8,2.8)
#fig.savefig(f"fig/resistivity_z_on_chi_PC_{chi_PS=}.svg")
#%%
