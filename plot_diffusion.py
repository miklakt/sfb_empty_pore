#%%
import numpy as np
from calculate_fields_in_pore import mobility_phi

import matplotlib.pyplot as plt

#%%
phi = np.geomspace(0.001, 0.7)
prefactor = np.array([1, 10, 100])
d = 8

D = [mobility_phi(phi, k=1, d=d, prefactor = _i) for _i in prefactor]

fig, ax = plt.subplots()
for prefactor_, D_ in zip(prefactor, D):
    ax.plot(phi, D_, label = prefactor_)

ax.set_xscale("log")
ax.set_xlabel("$\phi$")

ax.set_yscale("log")
ax.set_ylabel("$D/D_0$")

ax.legend(title = "prefactor")

# %%
