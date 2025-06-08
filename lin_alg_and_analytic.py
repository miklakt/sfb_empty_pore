#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib settings
LAST_USED_COLOR = lambda: plt.gca().lines[-1].get_color()
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
#%%
import R_lin_alg

def R_analytic(pore_radius:int, wall_thickness:int, d:int=None):
    if d is None:
        D_0 = 1.0
    else:
        if d>=2*pore_radius:
            raise ValueError("d > pore_radius")
        D_0 = 1/(3*np.pi*d)
    pore_radius_ = pore_radius-d/2
    wall_thickness_=wall_thickness+d
    R_int  = wall_thickness_/(D_0*np.pi*pore_radius_**2)
    R_ext = 1/(2*D_0*pore_radius_)
    R = R_int+R_ext
   
    return {"R_int":R_int, "R_ext":R_ext, "R":R}
#%%
pore_radius = 26
wall_thickness = 52
d_linalg= np.arange(2, 40,2)
d_linalg = [0.5, 1] + d_linalg.tolist()
d = d_linalg
analytic = [R_analytic(pore_radius, wall_thickness, d_) for  d_ in d]
analytic = pd.DataFrame(analytic)

linalg = [R_lin_alg.R_empty_pore(pore_radius, wall_thickness, d_, z_boundary=500) for d_ in d_linalg]
linalg = pd.DataFrame(linalg)
# %%
fig, ax = plt.subplots()
ax.set_xlabel("$d$")
ax.set_ylabel(r"$R \frac{k_\text{B} T}{\eta_{\text{S}}}$")
ax.set_xscale("log")
ax.set_yscale("log")

ax.plot(d, analytic["R"], label = "R", color = "k")
ax.plot(d, analytic["R_ext"], label = "R_ext", color = "k", linestyle="--")
ax.plot(d, analytic["R_int"], label = "R_int", color = "k", linestyle="-.")

ax.plot(d_linalg, linalg["R"], label = "R", color = "red")
#ax.plot(d_linalg, linalg["R_ext"], label = "R_ext", color = "red", linestyle="--")
ax.plot(d_linalg, linalg["R"] - analytic["R_int"], label = "R_ext", color = "green", linestyle="--")
ax.plot(d_linalg, linalg["R_int"], label = "R_int", color = "red", linestyle="-.")


ax.legend()

# %%
