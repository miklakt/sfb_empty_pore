import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import itertools

import numpy as np
import utils
import pandas as pd
from fit_gamma_model import gamma, free_energy_cylinder
#%%
s = 52
r = 26
ph =4
pw =ph
sigma = 0.02
chi_PS = [0.5]
chi_PC = [-1.5, -0.75, 0.0]

master = pd.read_pickle("pkl/reference_table.pkl")
#master = master.loc[master["comment"] == "grown_from_small"]
master_empty = pd.read_pickle("pkl/reference_table_empty_brush.pkl")
master_empty = master_empty.loc[(master_empty.s == s) & (master_empty.r== r) & (master_empty.sigma == sigma)]
master = master.loc[master.chi_PC.isin(chi_PC)]
master = master.loc[master.ph==ph]
master = master.loc[master.chi_PS.isin(chi_PS)]
master = master.loc[master.sigma == sigma]
#%%
gamma_f = gamma
fig, ax = plt.subplots()
gamma_f = gamma
for chi_PC_, df in master.groupby(by ="chi_PC"):
    empty_pore_data = utils.get_by_kwargs(master_empty, chi_PS = chi_PS)
    X = [1, 0]
    osm, sur = free_energy_cylinder(int(ph/2), empty_pore_data, chi_PS, chi_PC_, gamma_f, X)
    tot = osm+sur
    tot = tot[:len(tot)//2]
    x = list(range(-len(tot), 0))
    ax.plot(x, tot, 
        linestyle = "--",
        linewidth= 0.7,
        )
    X = [0.70585835, -0.31406453]
    osm, sur = free_energy_cylinder(int(ph/2), empty_pore_data, chi_PS, chi_PC_, gamma_f, X)
    tot = osm+sur
    tot = tot[:len(tot)//2]
    x = list(range(-len(tot), 0))
    ax.plot(x, tot, 
        color = ax.lines[-1].get_color(),
        label = "$\chi_{PC}=$" + str(chi_PC_)
        )
    df = df.loc[df.pc<=0]
    ax.scatter(df.pc, df.free_energy, marker = "o", s =20, facecolor = "none", edgecolor = ax.lines[-1].get_color())

ax.scatter([],[], marker = "o", s =20, facecolor = "none", edgecolor = 'black', label = "SF-SCF")
ax.plot([],[], linestyle = "--", linewidth= 0.7, color = "k", label = "$b_0 = 1.0, b_1 = 0.0$")
ax.plot([],[], color = "k", label = "$b_0 = 0.7, b_1 = -0.3$")

ax.axvline(-26, color = "green")


fig.legend(title = r"", bbox_to_anchor = [0.9,0.7], loc = "upper left")
ax.set_xlim(-60,0)

ax.set_xlabel("z")
ax.set_ylabel("$\Delta F / k_B T$")

fig.set_size_inches(4,3)
#fig.savefig("fig/fit.svg")