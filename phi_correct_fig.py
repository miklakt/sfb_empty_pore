#%%
import numpy as np
import matplotlib.pyplot as plt

a0, a1 = [0.70585835, -0.31406453]
chi_pc_ = np.linspace(-2, 0)
phi_correction_factor = (a0 + a1*chi_pc_)
fig, ax = plt.subplots()

ax.plot(chi_pc_, phi_correction_factor)
ax.set_xlabel("$\chi_{PC}$")
ax.set_ylabel("$\phi_{corrected}/\phi$")
ax.axhline(1, color = 'black')
# %%
phi_ = np.linspace(0, 0.7)
chi_pc_ = np.arange(-2, 0.5, 0.5)
chi = 0.5

gamma_ = [gamma(chi, chi_pc__, phi_, a0, a1) for chi_pc__ in chi_pc_]
fig, ax = plt.subplots()

[ax.plot(phi_, gamma__, label = chi_pc__) for chi_pc__, gamma__ in zip(chi_pc_, gamma_)]
ax.set_xlabel("$\phi$")
ax.set_ylabel("$\gamma$")
ax.axhline(0, color = 'black')
ax.legend(title="$\chi_{PC}$")