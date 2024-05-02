#%%
import numpy as np
from calculate_fields_in_pore import    mobility_Rubinstein, \
                                        mobility_Phillies, \
                                        mobility_FoxFlory, \
                                        mobility_Hoyst

import matplotlib.pyplot as plt

def mobility_Hoyst(phi, d, N, alpha, delta, nu = 0.76):
    R_g = np.sqrt(N/6)
    phi_entangled = (9/(2*np.pi))/R_g
    xi = R_g*(phi/phi_entangled)**(-nu)
    # if d>R_g:
    #     b = R_g/xi
    # else:
    b = d/xi
    D0_D =  np.exp(alpha*b**delta)
    return 1/D0_D

#%%
phi = np.geomspace(0.001, 0.7)
prefactor = np.array([1, 10, 100])
d = 8

D = [mobility_Rubinstein(phi, k=1, d=d, prefactor = _i) for _i in prefactor]

fig, ax = plt.subplots()
for prefactor_, D_ in zip(prefactor, D):
    ax.plot(phi, D_, label = prefactor_)

ax.set_xscale("log")
ax.set_xlabel("$\phi$")

ax.set_yscale("log")
ax.set_ylabel("$D/D_0$")

ax.legend(title = "prefactor")

# %%
fig, ax = plt.subplots()
phi = np.geomspace(0.001, 0.9)
d = 8

#D_Hoyst = mobility_Hoyst(phi, d = d, alpha=40, delta = 0.7)
D_Hoyst = mobility_Hoyst(phi, N=300, d = d, alpha=1.63, delta = 0.89)
D_Rubinstein = mobility_Rubinstein(phi, k=1, d=d, prefactor=1)
#D_Phillies = mobility_Phillies(phi, beta = 8, nu = 0.76)
#D_FoxFlory = mobility_FoxFlory(phi, N=300)

ax.plot(phi, D_Rubinstein, label = 'Rubinstein')
# ax.plot(phi, D_FoxFlory, label = 'Fox-Flory')
ax.plot(phi, D_Hoyst, label = 'Hoyst(empirical)')
# ax.plot(phi, D_Phillies, label = 'Phillies(empirical)')

ax.set_xscale("log")
ax.set_xlabel("$\phi$")

ax.set_yscale("log")
ax.set_ylabel("$D/D_0$")

ax.legend(title = "Polymer diffusion model")
# %%
