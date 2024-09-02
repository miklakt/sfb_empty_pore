#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib settings
#LAST_USED_COLOR = lambda: plt.gca().lines[-1].get_color()
#plt.rcParams['axes.xmargin'] = 0
#plt.rcParams['axes.ymargin'] = 0
#%%
def make_analytical_resistivity_on_z(pore_radius, wall_thickness, func_type = "elliptic"):
    def func(z:float)->float:
        if isinstance(z, (np.ndarray, list)): return np.array([func(_it) for _it in z])
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
#%%
pore_radius = 52
wall_thickness = 26
z = np.linspace(-100,100,1000)
d=8
r_z_empty = make_analytical_resistivity_on_z(pore_radius, wall_thickness)(z)
r_z_lower = make_analytical_resistivity_on_z(pore_radius-d/2, wall_thickness)(z)
r_z_upper = make_analytical_resistivity_on_z(pore_radius-d/2, wall_thickness+d)(z)
r_z_quad =  make_analytical_resistivity_on_z(pore_radius-d/2, wall_thickness, func_type = "quad")(z)
#r_z_empty_quad = make_analytical_resistivity_on_z(pore_radius, wall_thickness)(z)
#%%
fig,ax =plt.subplots()
ax.plot(z, r_z_empty)
ax.plot(z, r_z_lower)
ax.plot(z, r_z_upper)
ax.plot(z, r_z_quad)
# %%
