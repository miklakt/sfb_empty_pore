# %%
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rc

#rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')
style.use('tableau-colorblind10')
mpl_markers = ('o', '+', 'x', 's', 'D')

from calculate_fields_in_pore import *
#%%
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

def quadratic_spline_find_x(X, Y, y_value):
    # Interpolating using quadratic spline (degree 2)
    quadratic_spline = interp1d(X, np.log(Y), kind='quadratic', bounds_error=False, fill_value="extrapolate")
    
    # Function to find the root where the quadratic spline matches y_value
    def find_x_for_y(x_guess):
        return quadratic_spline(x_guess) - np.log(y_value)
    
    # Provide an initial guess for fsolve, usually in the range of the X values
    x_initial_guess = np.mean(X)
    
    # Solve for x
    x_solution = fsolve(find_x_for_y, x_initial_guess)
    
    # Check if the solution is within the bounds of the input X array
    if X.min() <= x_solution[0] <= X.max():
        return x_solution[0]
    else:
        return None
#%%

a0 = 0.70585835
a1 = -0.31406453
wall_thickness=52
pore_radius=26
sigma = 0.02
#%%
#%%
d_color= [4, 8, 12, 16, 20]
d = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
#d =[8 ,10, 12 ,]
chi_PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#chi_PC = [-2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1, -0.75]
chi_PC = np.round(np.arange(0, -2.6, -1.05),3)

# model, mobility_model_kwargs = "none", {}
# model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
# model = "Fox-Flory", dict(N = 300)
model, mobility_model_kwargs = "Rubinstein", {"prefactor":1}
#model, mobility_model_kwargs = "Hoyst", {"alpha" : 1.63, "delta": 0.89, "N" : 300}

results = []
for d_, chi_PS_, chi_PC_ in itertools.product(d, chi_PS, chi_PC):
    print(d_, chi_PS_, chi_PC_)
    result = calculate_permeability(
        a0, a1, pore_radius, wall_thickness,
        d_, chi_PS_, chi_PC_,
        sigma = sigma,
        exclude_volume=True,
        truncate_pressure=False,
        method= "convolve",
        convolve_mode="same",
        mobility_correction= "vol_average",
        mobility_model = model,
        mobility_model_kwargs = mobility_model_kwargs,
        integration="cylindrical_caps"
        )
        
    #result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
    results.append(result)
results = pd.DataFrame(results)
#%%
def empty_pore_permeability(D, r, s):
    #einstein_factor = 1/(3*np.pi*d)
    return 2*D*r/(1 + 2*s/(r*np.pi))# / einstein_factor
#%%
R_empty_dict = {}
chi_PC_crit_dict = {}
for chi_PS_, result_ in  results_.groupby(by = "chi"):
    R_empty = []
    chi_PC_crit = []
    for d_, result__ in result_.groupby(by = "d"):
        x = result__["chi_PC"].squeeze()
        y = 1/result__["permeability"]
        R_empty_ = 1/result__["thick_empty_pore"].iloc[0]
        X0_ = quadratic_spline_find_x(x, y, R_empty_)

        R_empty.append(R_empty_)
        chi_PC_crit.append(X0_)
    R_empty_dict[chi_PS_] = R_empty
    chi_PC_crit_dict[chi_PS_] = chi_PC_crit

#%%
fig, axs = plt.subplots(ncols = len(chi_PS), sharey="row", nrows = 1, sharex = True)
results_ = results.loc[(results.mobility_model == model)]

for ax, (chi_PS_, result_) in zip(axs, results_.groupby(by = "chi")):
    markers = itertools.cycle(mpl_markers)
    for d_, result__ in result_.groupby(by = "d"):
        x = result__["chi_PC"].squeeze()
        y = 1/result__["permeability"]


        # y = np.gradient(np.log(y))/np.gradient(np.log(x))
        # y = np.gradient(y)/np.gradient(np.log(x))
        if  d_>24:continue

        if d_ in d_color:
            plot_kwargs = dict(
                label = fr"$d = {d_}$",
                #marker = next(markers),
                #markevery = 0.5,
                #markersize = 4,
                linewidth = 0.7,
                #color ="darkorange"
                color = "black"
            )
        else:
            plot_kwargs = dict(
                linewidth = 0.3,
                color ="black"
            )
        ax.plot(
            x, y, 
            **plot_kwargs
            )

        # ax.axhline(
        #     empty_pore_permeability(1, pore_radius-d_/2, wall_thickness+d_) * (3*np.pi*d_), 
        #     color = ax.lines[-1].get_color()
        #     #color = "red"
        #     )

        # ax.scatter([X0], [R_empty], color = ax.lines[-1].get_color(), marker = "x")
    ax.plot(
        chi_PC_crit_dict[chi_PS_], 
        R_empty_dict[chi_PS_],
        #color = "red",
        #linestyle = "--",
        #label = "$R_{empty}$",
        linewidth = 2,
        zorder = 3,
        #marker = "x"
        )
        


    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    ax.set_ylim(5e-1, 1e3)
    ax.set_ylim(-2,0)
    ax.set_yscale("log")
    #ax.set_xscale("log")

#axs[0].legend()

#axs[0].set_ylabel(r"$R \cdot \frac{k_B T}{\eta_0}$")
# axs[1,-1].plot([],[], color = "black", linestyle = ":", linewidth = 2,

#                 label = "$R_{conv}$"
#                 )

# axs[1,-1].plot([],[], color = "black", linestyle = "-.", linewidth = 2,
#                 label = "$R_{channel}$"
#                 )

# axs[1,-1].plot([],[], color = "black", linestyle = "-", linewidth = 2,
#                 label = "$R_{empty}$"
#                 )
#fig.supxlabel("$\chi_{PC}$")
    #axs[1,-1].legend()

#plt.tight_layout()
#fig.set_size_inches(7, 7)
fig.set_size_inches(7, 2.5)
#fig.savefig("fig/resistivity_on_d.svg")
#fig.savefig("tex/third_report/fig/permeability_on_d_detailed_low_d.svg")
# %%
fig, ax = plt.subplots()

markers = itertools.cycle(mpl_markers)
for chi_PS_, result_ in results_.groupby(by = "chi"):
    marker = next(markers)
    ax.plot(
        d,
        chi_PC_crit_dict[chi_PS_], 
        #color = "red",
        #linestyle = "--",
        label = chi_PS_,
        linewidth = 2,
        zorder = 3,
        marker = "o",
        mfc = "none",
        markersize = 4,
    )
ax.set_xlabel("$d$")
ax.set_ylabel(r"$\chi_{\text{PC}}^{\text{crit}}$")
ax.legend(title= r"$\chi_{\text{PS}}$")
#ax.set_xscale("log")
fig.set_size_inches(3, 3)
fig.savefig("fig/chi_PC_crit_on_d.svg")
# %%
