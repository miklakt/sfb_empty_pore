# %%
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rc

rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')
style.use('tableau-colorblind10')
mpl_markers = ('o', '+', 'x', 's', 'D')

from calculate_fields_in_pore import *
from pickle_cache import pickle_lru_cache

a0 = 0.70585835
a1 = -0.31406453
wall_thickness=52
pore_radius=26
sigma = 0.02




model_formula = dict(
    Rubinstein = r"$\frac{D}{D_0} = \frac{(f \phi d)^{-2}}{1+(f \phi d)^{-2}}$",
    Hoyst = r"$\frac{D}{D_0} = \exp \left( -\alpha \left( \frac{d}{\xi} \right)^{\delta}  \right)$"
)

einstein_visc  = r"$D_0 = \frac{k_B T}{3 \pi \eta d}$"
perm_formula = \
    r"$P_R \cdot \frac{\eta_0}{k_B T} = {\frac{1}{3 d}} \left( \frac{\pi}{2(r-d/2)} + \frac{s}{(r-d/2)^2} \right)^{-1}$" 
perm_int = \
    r"$P_{\int} \cdot \frac{\eta_0}{k_B T} = \frac{2}{3 d} \left[\int_{-s/2-l_R/2}^{s/2+l_R/2} \left( \int_{0}^{r_{pore}} \frac{D}{D_0} e^{-\Delta F / kT} r dr \right)^{-1} dz \right]^{-1}$"
rayleigh_length = r"$l_R = \frac{\pi}{2} r_{pore}$"
#%%
simulation_results = pd.DataFrame(
   columns =    [   "d",   "chi_PS",    "chi_PC",   "J_tot",    "J_tot_err"],
   data = [     [   4,      0.5,        -1.0,       4.936,      0.014],
                [   4,      0.5,        -1.25,      7.716,      0.018],
                [   6,      0.5,        -1.25,     10.596,      0.028],
                [   8,      0.5,        -1.25,     16.131,      0.042],
                [  12,      0.5,        -1.25,     31.022,      0.080],
                [  16,      0.5,        -1.25,     38.650,      0.376],
                [  24,      0.5,        -1.25,      27.00,      10.00],
                [   4,      0.5,        -1.0,       4.933,      0.011],
                [   6,      0.5,        -1.0,       3.943,      0.010],
                [   8,      0.5,        -1.0,       2.954,      0.007],
                [  12,      0.5,        -1.0,       0.844,      0.002],
                [   4,      0.4,        -1.25,      6.756,      0.016],
                [   6,      0.4,        -1.25,      7.065,      0.017],
                [   8,      0.4,        -1.25,      7.145,      0.017],
                [  10,      0.4,        -1.25,      6.118,      0.015],
                [  12,      0.4,        -1.25,      3.731,      0.010],
                [   8,      0.4,         -1.0,      1.392,      0.004],
                [   4,      0.4,         -1.5,     10.037,      0.023],
                [   8,      0.4,         -1.5,     25.761,      0.060],
                [   4,      0.3,         -1.5,      8.562,      0.021],
                [   8,      0.3,         -1.5,     14.102,      0.034],
                [  12,      0.3,         -1.5,     14.362,      0.036],
                [  14,      0.3,         -1.5,      7.362,      0.036],
                [  16,      0.3,         -1.5,      1.898,      0.007],
                [  10,      0.6,         -1.0,     11.168,      0.029]
   ]
)



simulation_empty_pore = pd.DataFrame(
    columns = ["d", "J_tot"],
    data = dict(
            d=[4, 8, 16, 24],
            #R = [0.203, 0.302]
            J_tot = [5.927, 4.926, 3.311, 2.063]
        )
)

simulation_empty_pore_half = pd.DataFrame(
    columns = ["d", "J_tot"],
    data = dict(
            d   =   [2,     4,      8,      16,     24  ],
            #R = [0.203, 0.302]
            J_tot = [12.89, 11.91,  9.99,   6.72,   4.16]
        )
)
#%%
d = np.arange(2, 34, 2)
#d =[8 ,10, 12 ,]
chi_PS = [0.4, 0.5, 0.7]
chi_PC = [-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5]
chi_PC_color = [-1.5, -1.25, -1, -0.75]
L=20

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
        d_, chi_PS_, chi_PC_, L,
        sigma = sigma,
        exclude_volume=True,
        truncate_pressure=False,
        method= "convolve", 
        mobility_correction= "vol_average",
        mobility_model = model,
        mobility_model_kwargs = mobility_model_kwargs
        )
        
    result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)
    results.append(result)
results = pd.DataFrame(results)

#%%
fig, axs = plt.subplots(ncols = len(chi_PS), sharey=True)
results_ = results.loc[(results.mobility_model == model)]

permeability_field = "limited_permeability"
for ax, (chi_PS_, result_) in zip(axs, results_.groupby(by = "chi")):
    markers = itertools.a0 = 0.70585835lt__ in result_.groupby(by="chi_PC"):
        x = result__["d"].squeeze()
        y = result__[permeability_field].squeeze()
        if chi_PC_ in chi_PC_color:
            plot_kwargs = dict(
                label = fr"$P_{{\int}}\left(\chi_{{PC}} = {chi_PC_}\right)$",
                marker = next(markers),
                markevery = 0.2
            )
        else:
            plot_kwargs = dict(
                linewidth = 0.1,
                color ="black"
            )
        ax.plot(
            x, y, 
            **plot_kwargs
            )

        # R = simulation_results.query(f"(chi_PS=={chi_PS_})&(chi_PC=={chi_PC_})")
        # ax.scatter(
        #     R["d"], R["J_tot"]/R["d"]/3, 
        #     color = ax.lines[-1].get_color(),
        #     marker = "o",
        #     s = 50,
        #     facecolor = "none"
        #     )

    ax.plot(
        d, 
        result__["thick_empty_pore"], 
        color = "black", 
        linestyle = "--",
        label = "$P_{R}(d)$"
        )

    ax.plot(
        d, 
        result__["thin_empty_pore"], 
        color = "black", 
        linestyle = ":",
        label = "$P_{R}(s=0)$"
        )

    # ax.plot(
    #     d, 
    #     result__["thin_empty_pore"]*4, 
    #     color = "black", 
    #     linestyle = ":",
    #     label = "$P_{half-pore}$"
    #     )


    # ax.scatter(
    #     simulation_empty_pore["d"], 
    #     simulation_empty_pore["J_tot"]/simulation_empty_pore["d"]/3,
    #     color = "black",
    #     marker = "o",
    #     label = "simulation",
    #     s=50,
    #     )

    # ax.scatter(
    #     simulation_empty_pore_half["d"], 
    #     simulation_empty_pore_half["J_tot"]/simulation_empty_pore_half["d"]/3,
    #     color = "black",
    #     marker = "o",
    #     label = "simulation",
    #     s=50,
    #     )

    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel("d")
    ax.set_yscale("log")
    ax.set_xscale("log")

axs[0].set_ylabel(r"$P \cdot \frac{\eta_0}{k_B T} $")
axs[-1].legend( 
    bbox_to_anchor = [1.0, -0.05],
    loc = "lower left"
    )

text = " ".join([model_formula[model],einstein_visc, rayleigh_length]) + "\n" +\
     "\n".join([perm_formula, perm_int])

fig.text(s = text, x = 0.72, y = 0.95, ha = "left", va = "top", fontsize = 14)
plt.tight_layout()
fig.set_size_inches(10, 3.5)

# %%


#%%
## Figure 1 
## Permeability on particle size at different solvent quality and particle affinity
a0 = 0.70585835
a1 = -0.31406453
wall_thickness=52
pore_radius=26
d = np.arange(2, 34, 2)
#d =[8 ,10, 12 ,]
chi_PS = [0.4, 0.5, 0.6]
chi_PC = -1.25
L=20

# model, mobility_model_kwargs = "none", {}
# model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
# model = "Fox-Flory", dict(N = 300)
model, prefactors = "Rubinstein", [0, 1, 1e1, 1e2]
#model, alphas, delta = "Hoyst", [0, 0.5, 1.5, 4.5], 0.89

results = []
if model == "Rubinstein":
    for d_, chi_PS_, prefactor_ in itertools.product(d, chi_PS, prefactors):
        result = calculate_permeability(
            a0, a1, pore_radius, wall_thickness,
            d_, chi_PS_, chi_PC, L,
            exclude_volume=True,
            truncate_pressure=False,
            method= "convolve", 
            mobility_correction= "vol_average",
            mobility_model = model,
            mobility_model_kwargs = {"prefactor" : prefactor_}
            )
        results.append(result)
    results = pd.DataFrame(results)
    results["prefactor"] = results.apply(lambda _: _["mobility_model_kwargs"]["prefactor"], axis = 1)
if model == "Hoyst":
    for d_, chi_PS_, alpha_ in itertools.product(d, chi_PS, alphas):
        result = calculate_permeability(
            a0, a1, pore_radius, wall_thickness,
            d_, chi_PS_, chi_PC, L,
            exclude_volume=True,
            truncate_pressure=False,
            method= "convolve", 
            mobility_correction= "vol_average",
            mobility_model = model,
            mobility_model_kwargs = {"alpha" : alpha_, "delta" : delta, "N":300}
            )
        results.append(result)
    results = pd.DataFrame(results)
    results["alpha"] = results.apply(lambda _: _["mobility_model_kwargs"]["alpha"], axis = 1)
# %%

markers = itertools.cycle(mpl_markers)
fig, axs = plt.subplots(ncols = len(chi_PS), sharey=True)
results_ = results.loc[(results.mobility_model == model)]


for ax, (chi_PS_, result_) in zip(axs, results_.groupby(by = "chi")):
    markers = itertools.cycle(mpl_markers)
    group_column = "prefactor" if model == "Rubinstein" else "alpha"
    label = "prefactor" if model == "Rubinstein" else r"\alpha"
    #for prefactor_, result__ in result_.groupby(by="prefactor"):
    for it_, result__ in result_.groupby(by=group_column):
        x = result__["d"].squeeze()
        y = result__["permeability"].squeeze()
        plot_kwargs = dict(
            #label = fr"$f = {prefactor_}$",
            label = fr"${label} = {it_}$",
            marker = next(markers),
            markevery = 0.2
        )
        ax.plot(
            x, y, 
            **plot_kwargs
            )

    ax.plot(
        d, 
        result__["thick_empty_pore"], 
        color = "black", 
        linestyle = "--",
        label = "$P_{R}(d)$"
        )

    ax.plot(
        d, 
        result__["thin_empty_pore"], 
        color = "black", 
        linestyle = ":",
        label = "$P_{R}(s=0)$"
        )

    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel("d")

    ax.set_yscale("log")
    ax.set_xscale("log")

axs[0].set_ylabel(r"$P \cdot \frac{\eta_0}{k_B T} $")
axs[-1].legend( 
    bbox_to_anchor = [1.0, -0.05],
    loc = "lower left"
    )

text = " ".join([model_formula[model], einstein_visc, rayleigh_length]) + "\n" +\
     "\n".join([perm_formula, perm_int])

fig.text(s = text, x = 0.8, y = 0.92, ha = "left", va = "top", fontsize = 14)
fig.text(s = f"$\chi_{{PC}} = {chi_PC}$", x = 0.15, y = 0.85, ha = "left", va = "top", fontsize = 16)

plt.tight_layout()
fig.set_size_inches(10, 3.5)
# %%
model, alpha, deltas = "Hoyst", 1.63, [0.2, 0.8, 1.4, 2.0]

results = []
if model == "Rubinstein":
    for d_, chi_PS_, prefactor_ in itertools.product(d, chi_PS, prefactors):
        result = calculate_permeability(
            a0, a1, pore_radius, wall_thickness,
            d_, chi_PS_, chi_PC, L,
            exclude_volume=True,
            truncate_pressure=False,
            method= "convolve", 
            mobility_correction= "vol_average",
            mobility_model = model,
            mobility_model_kwargs = {"prefactor" : prefactor_}
            )
        results.append(result)
    results = pd.DataFrame(results)
    results["prefactor"] = results.apply(lambda _: _["mobility_model_kwargs"]["prefactor"], axis = 1)
if model == "Hoyst":
    for d_, chi_PS_, delta_ in itertools.product(d, chi_PS, deltas):
        result = calculate_permeability(
            a0, a1, pore_radius, wall_thickness,
            d_, chi_PS_, chi_PC, L,
            exclude_volume=True,
            truncate_pressure=False,
            method= "convolve", 
            mobility_correction= "vol_average",
            mobility_model = model,
            mobility_model_kwargs = {"alpha" : alpha, "delta" : delta_, "N":300}
            )
        results.append(result)
    results = pd.DataFrame(results)
    results["delta"] = results.apply(lambda _: _["mobility_model_kwargs"]["delta"], axis = 1)
# %%

markers = itertools.cycle(mpl_markers)
fig, axs = plt.subplots(ncols = len(chi_PS), sharey=True)
results_ = results.loc[(results.mobility_model == model)]


for ax, (chi_PS_, result_) in zip(axs, results_.groupby(by = "chi")):
    markers = itertools.cycle(mpl_markers)
    for delta_, result__ in result_.groupby(by="delta"):
        x = result__["d"].squeeze()
        y = result__["permeability"].squeeze()
        plot_kwargs = dict(
            label = fr"$\delta = {delta_}$",
            marker = next(markers),
            markevery = 0.2
        )
        ax.plot(
            x, y, 
            **plot_kwargs
            )

    ax.plot(
        d, 
        result__["thick_empty_pore"], 
        color = "black", 
        linestyle = "--",
        label = "$P_{R}(d)$"
        )

    ax.plot(
        d, 
        result__["thin_empty_pore"], 
        color = "black", 
        linestyle = ":",
        label = "$P_{R}(s=0)$"
        )

    ax.set_title(f"$\chi_{{PS}} = {chi_PS_}$")
    ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel("d")

    ax.set_yscale("log")
    ax.set_xscale("log")

axs[0].set_ylabel(r"$P \cdot \frac{\eta_0}{k_B T} $")
axs[-1].legend( 
    bbox_to_anchor = [1.0, -0.05],
    loc = "lower left"
    )

text = " ".join([model_formula[model], einstein_visc, rayleigh_length]) + "\n" +\
     "\n".join([perm_formula, perm_int])

fig.text(s = text, x = 0.8, y = 0.92, ha = "left", va = "top", fontsize = 14)
fig.text(s = f"$\chi_{{PC}} = {chi_PC}$", x = 0.15, y = 0.85, ha = "left", va = "top", fontsize = 16)

plt.tight_layout()
fig.set_size_inches(10, 3.5)
# %%
a0 = 0.70585835
a1 = -0.31406453
wall_thickness=52
pore_radius=26
d = np.arange(2, 34, 2)
#d =[8 ,10, 12 ,]
chi_PS = 0.5
chi_PC = [-1.0, -1.25, -1.5]
L=[0, 20]

# model, mobility_model_kwargs = "none", {}
# model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
# model = "Fox-Flory", dict(N = 300)
#model, prefactor = "Rubinstein", [0]

#model, alphas, delta = "Hoyst", [0, 0.5, 1.5, 4.5], 0.89

results = []

for d_, chi_PC_, L_ in itertools.product(d, chi_PC, L):
    result = calculate_permeability(
        a0, a1, pore_radius, wall_thickness,
        d_, chi_PS, chi_PC_, L_,
        exclude_volume=True,
        truncate_pressure=False,
        method= "convolve", 
        mobility_correction= "vol_average",
        mobility_model = "Rubinstein",
        mobility_model_kwargs = {"prefactor" : 1}
        )
    result["comment"] = "Diffusion_and_energy"
    results.append(result)
    result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)


for d_, chi_PC_ in itertools.product(d, chi_PC):
    result = calculate_permeability(
        a0, a1, pore_radius, wall_thickness,
        d_, chi_PS, chi_PC_, L=20,
        exclude_volume=True,
        truncate_pressure=False,
        method= "no_free_energy", 
        mobility_correction= "vol_average",
        mobility_model = "Rubinstein",
        mobility_model_kwargs = {"prefactor" : 1}
        )
    result["comment"] = "Diffusion_only"
    results.append(result)

for d_, chi_PC_ in itertools.product(d, chi_PC):

    result = calculate_permeability(
        a0, a1, pore_radius, wall_thickness,
        d_, chi_PS, chi_PC_, L=20,
        exclude_volume=True,
        truncate_pressure=False,
        method= "convolve", 
        mobility_correction= "vol_average",
        mobility_model = "Rubinstein",
        mobility_model_kwargs = {"prefactor":0}
        )
    result["comment"] = "Free_energy_only"
    results.append(result)
results = pd.DataFrame(results)
#results["prefactor"] = results.apply(lambda _: _["mobility_model_kwargs"]["prefactor"], axis = 1)

# %%
markers = itertools.cycle(mpl_markers)
fig, axs = plt.subplots(ncols = len(chi_PC), sharey=True)

results_ = results.loc[(results["comment"] == "Diffusion_and_energy") & (results["L"] == 20)]

for ax, (chi_PC_, result_) in zip(axs, results_.groupby(by = "chi_PC")):
    markers = itertools.cycle(mpl_markers)
    for L_, result__ in result_.groupby(by="L"): 
        x = result__["d"].squeeze()
        y = result__["permeability"].squeeze()
        plot_kwargs = dict(
            label = fr"$l_{{R}} = {L_}$",
            #marker = next(markers),
            #markevery = 0.2,
            color = "tab:orange",
            linestyle = "solid" if L_==20 else "--"
        )
        ax.plot(
            x, y, 
            **plot_kwargs
            )

    ax.plot(
        d, 
        result__["thick_empty_pore"], 
        color = "black", 
        linestyle = "--",
        label = "$P_{R}(d)$"
        )

    ax.plot(
        d, 
        result__["thin_empty_pore"], 
        color = "black", 
        linestyle = ":",
        label = "$P_{R}(s=0)$"
        )

    ax.set_title(f"$\chi_{{PC}} = {chi_PC_}$")
    ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel("d")

    ax.set_yscale("log")
    ax.set_xscale("log")


################################################################################
results_ = results.loc[results["comment"] == "Diffusion_only"]
for ax, (chi_PC_, result_) in zip(axs, results_.groupby(by = "chi_PC")):
    markers = itertools.cycle(mpl_markers)
    for L_, result__ in result_.groupby(by="L"): 
        x = result__["d"].squeeze()
        y = result__["permeability"].squeeze()
        plot_kwargs = dict(
            label = fr"$\Delta F = 0$",
            #marker = next(markers),
            #markevery = 0.2,
            color = "black",
            linewidth = 2
        )
        ax.plot(
            x, y, 
            **plot_kwargs
            )


################################################################################
# results_ = results.loc[results["comment"] == "Free_energy_only"]
# for ax, (chi_PC_, result_) in zip(axs, results_.groupby(by = "chi_PC")):
#     markers = itertools.cycle(mpl_markers)
#     for L_, result__ in result_.groupby(by="L"): 
#         x = result__["d"].squeeze()
#         y = result__["permeability"].squeeze()
#         plot_kwargs = dict(
#             label = r"$\frac{D}{D_0} = 1$",
#             #marker = next(markers),
#             #markevery = 0.2,
#             color = "red",
#             linewidth = 2
#         )
#         ax.plot(
#             x, y, 
#             **plot_kwargs
#             )

results_ = results.loc[(results["comment"] ==  "Diffusion_and_energy") & (results["L"] == 0)]
for ax, (chi_PC_, result_) in zip(axs, results_.groupby(by = "chi_PC")):
    markers = itertools.cycle(mpl_markers)
    x = result_["d"].squeeze()
    y = result_["limited_permeability"].squeeze()
    plot_kwargs = dict(
        label = r"convergent flow limit",
        #marker = next(markers),
        #markevery = 0.2,
        color = "green",
        linewidth = 2
    )
    ax.plot(
        x, y, 
        **plot_kwargs
        )








axs[0].set_ylabel(r"$P \cdot \frac{\eta_0}{k_B T} $")
axs[-1].legend( 
    bbox_to_anchor = [1.0, 0.35],
    loc = "lower left"
    )

# text = " ".join([model_formula[model], einstein_visc, rayleigh_length]) + "\n" +\
#      "\n".join([perm_formula, perm_int])
text = fr"$\chi_{{PS}} = {chi_PS}$"
fig.text(s = text, x = 0.71, y = 0.92, ha = "left", va = "top", fontsize = 14)
#fig.text(s = f"$\chi_{{PC}} = {chi_PC}$", x = 0.15, y = 0.85, ha = "left", va = "top", fontsize = 16)

plt.tight_layout()
fig.set_size_inches(10, 3.5)

fig.savefig("fig.pdf")
# %%
a0 = 0.70585835
a1 = -0.31406453
wall_thickness=52
pore_radius=26
d = np.arange(2, 34, 2)
#d =[8 ,10, 12 ,]
chi_PS = 0.5
chi_PC = [-1.0, -1.25, -1.5]
L=20
sigmas = [0.004, 0.006, 0.008, 0.01, 0.014, 0.016, 0.02, 0.024, 0.026, 0.03]
sigmas_colored = [0.004, 0.01, 0.02, 0.03]

# model, mobility_model_kwargs = "none", {}
# model, mobility_model_kwargs = "Phillies", dict(beta = 8, nu = 0.76)
# model = "Fox-Flory", dict(N = 300)
#model, prefactor = "Rubinstein", [0]

#model, alphas, delta = "Hoyst", [0, 0.5, 1.5, 4.5], 0.89

results = []

for d_, chi_PC_, sigma_ in itertools.product(d, chi_PC, sigmas):
    result = calculate_permeability(
        a0, a1, pore_radius, wall_thickness,
        d_, chi_PS, chi_PC_, L,
        sigma = sigma_,
        exclude_volume=True,
        truncate_pressure=False,
        method= "convolve", 
        mobility_correction= "vol_average",
        mobility_model = "Rubinstein",
        mobility_model_kwargs = {"prefactor" : 1},
        #mobility_model_kwargs = {}
        )
    #result["comment"] = "Diffusion_and_energy"
    result["limited_permeability"] = (result["permeability"]**(-1) + result["thin_empty_pore"]**(-1))**(-1)

    fields = calculate_fields(
        a0=a0, a1=a1, d=d_, sigma = sigma_,
        chi_PC=chi_PC_, chi=chi_PS,
        wall_thickness=wall_thickness,
        pore_radius=pore_radius,
        exclude_volume=True,
        truncate_pressure=False,
        method= "convolve", 
        mobility_correction= "vol_average",
        mobility_model = "Rubinstein",
        mobility_model_kwargs = {"prefactor" : 1},
        #mobility_model_kwargs = {}
        )
    result["phi"] = fields["phi"].squeeze()[:,0]
    result["free_energy"] = fields["free_energy"].squeeze()[:,0]

    results.append(result)

results = pd.DataFrame(results)
#results["prefactor"] = results.apply(lambda _: _["mobility_model_kwargs"]["prefactor"], axis = 1)

# %%
markers = itertools.cycle(mpl_markers)
fig, axs = plt.subplots(ncols = len(chi_PC), sharey=True)

results_ = results

for ax, (chi_PC_, result_) in zip(axs, results_.groupby(by = "chi_PC")):
    markers = itertools.cycle(mpl_markers)
    for sigma__, result__ in result_.groupby(by="sigma"): 
        x = result__["d"].squeeze()
        y = result__["permeability"].squeeze()
        plot_kwargs = dict(
            label = fr"$\sigma = {sigma__}$",
            marker = next(markers) if sigma__ in sigmas_colored else None,
            markevery = 0.2,
            color = None if sigma__ in sigmas_colored else "grey",
            linewidth = 2 if sigma__ in sigmas_colored else 0.4,
            #linestyle = "solid" if L_==20 else "--"
        )
        ax.plot(
            x, y, 
            **plot_kwargs
            )

    ax.plot(
        d, 
        result__["thick_empty_pore"], 
        color = "black", 
        linestyle = "--",
        label = "$P_{R}(d)$"
        )

    ax.plot(
        d, 
        result__["thin_empty_pore"], 
        color = "black", 
        linestyle = ":",
        label = "$P_{R}(s=0)$"
        )

    ax.set_title(f"$\chi_{{PC}} = {chi_PC_}$")
    ax.set_ylim(1e-3, 1e3)
    ax.set_xlabel("d")

    ax.set_yscale("log")
    ax.set_xscale("log")


axs[0].set_ylabel(r"$P \cdot \frac{\eta_0}{k_B T} $")
axs[-1].legend( 
    bbox_to_anchor = [1.0, 0.35],
    loc = "lower left"
    )

# text = " ".join([model_formula[model], einstein_visc, rayleigh_length]) + "\n" +\
#      "\n".join([perm_formula, perm_int])
text = fr"$\chi_{{PS}} = {chi_PS}$"
fig.text(s = text, x = 0.71, y = 0.92, ha = "left", va = "top", fontsize = 14)
#fig.text(s = f"$\chi_{{PC}} = {chi_PC}$", x = 0.15, y = 0.85, ha = "left", va = "top", fontsize = 16)

plt.tight_layout()
fig.set_size_inches(10, 3.5)

fig.savefig("fig.pdf")
# %%
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches

markers = itertools.cycle(mpl_markers)
fig, axs = plt.subplots(ncols = len(chi_PC), sharey=True)
markers = itertools.cycle(mpl_markers)

results_ = results.loc[results.d == 16]

for ax, (chi_PC_, result_) in zip(axs, results_.groupby(by = "chi_PC")):
    markers = itertools.cycle(mpl_markers)
    trans = transforms.blended_transform_factory(
        ax.transData, ax.transAxes
        )
    for sigma__, result__ in result_.groupby(by="sigma"): 
        y = result__["free_energy"].squeeze()
        x = np.arange(len(y)) - len(y)/2
        plot_kwargs = dict(
            label = fr"$\sigma = {sigma__}$",
            marker = next(markers) if sigma__ in sigmas_colored else None,
            markevery = 0.2,
            color = None if sigma__ in sigmas_colored else "grey",
            linewidth = 2 if sigma__ in sigmas_colored else 0.4,
            #linestyle = "solid" if L_==20 else "--"
        )
        ax.plot(
            x, y, 
            **plot_kwargs
            )
    
    rect = mpatches.Rectangle((-wall_thickness/2, 0), width=wall_thickness, height=1, transform=trans,
                            color='grey', alpha=0.1)
    ax.add_patch(rect)
    #ax.legend(title="$\chi_{PC}$")
    ax.set_xlim(-80,80)
    #ax.set_ylim(-0.02, 0.3)
    ax.set_title(f"$\chi_{{PC}} = {chi_PC_}$")

    ax.set_xlabel("$z$")
axs[0].set_ylabel("$F(z, r=0)$")

axs[-1].legend( 
    bbox_to_anchor = [1.0, 0.0],
    loc = "lower left"
    )

fig.set_size_inches(3.5, 3)
# %%
