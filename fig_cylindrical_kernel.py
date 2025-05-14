#%%
import matplotlib.pyplot as plt
import numpy as np
from fit_gamma_model import cylinder_r0_kernel
volume, surface = cylinder_r0_kernel(8, 16)
fig, ax = plt.subplots(nrows = 2, sharex = True)
fig.set_size_inches(4,4.5)

ax[0].pcolormesh(volume, edgecolor = "black", linewidth = 0.7)
ax[0].axis("equal")

ax[1].pcolormesh(surface, edgecolor = "black", linewidth = 0.7)
ax[1].axis("equal")

ax[1].set_xticks(np.arange(0,18, 2))

#fig.savefig("/fug/cylindrical_kernel.svg")

# %%
