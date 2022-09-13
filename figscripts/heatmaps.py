#%%
import sys
import numpy
sys.path.append('..')

import pandas as pd
from utils import plot_imshow, get_by_kwargs
from theory import free_energy_phi

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

#matplotlib settings
LAST_USED_COLOR = lambda: plt.gca().lines[-1].get_color()
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


#%%
df = pd.read_pickle("../data/empty_brush.pkl")
#%%
for chi in [0.9]:
    fig = plot_imshow(
        df, 
        "phi", 
        chi_PS = chi, 
        s=52, 
        r=26, 
        contour = True,
        #colorbar=False,
        patches_kwargs=dict(
            linewidth = 0,
            facecolor = "darkgreen"
            ),
        imshow_kwargs = dict(
            cmap = "RdBu_r",
            #vmin = 0,
            #vmax = 0.45
        ),
    )
    fig.savefig(f"phi_2d_{chi}.pdf")
#%%
free_energy_phi_func = numpy.vectorize(free_energy_phi, excluded=["a1", "a2", "chi_PS", "chi_PC", "w", "h"])
chi_PC= -3.0
d=4
df["fe"] = df.apply(lambda _: free_energy_phi_func(phi =_.phi, a1=0.16, a2=0.08, chi_PC=chi_PC, chi_PS=_.chi_PS, w=d), axis =1)
fig = plot_imshow(
    df, 
    "fe", 
    chi_PS = 0.8, 
    s=52, 
    r=26, 
    contour=True, 
    patches_kwargs=dict(
        linewidth = 0,
        facecolor = "darkgreen"
        ),
    imshow_kwargs = dict(
        cmap = "RdBu",
        #vmin = -12,
        vmax = -15
    )
)
fig.savefig(f"fe_2d_{chi}_{chi_PC}_{d}.pdf")
# %%
