#%%
import matplotlib.pyplot as plt
import numpy as np

X = [0.7, -0.3]
chi_crit = 6*np.log(5/6)
chi_pc_crit = (1-X[0])/X[1]

fig, ax = plt.subplots()

CHI_PC = np.linspace(-1.5, 0, 50)
F = CHI_PC*X[1] + X[0]

ax.plot(CHI_PC, F)
ax.axhline(1, color = "black")
#ax.axvline(chi_pc_crit, color = "black", linestyle = "--")
ax.text(chi_pc_crit, 1, fr"$\chi^{{*}}_{{PC}} = {chi_pc_crit:.2f}$", va = "bottom")
ax.set_xlabel(r"$\chi_{PC}$")
ax.set_ylabel(r"$\phi^{*} / \phi$")
fig.set_size_inches(3,3)
plt.tight_layout()
#fig.savefig("/fig/phi_correction.png", dpi =600)
# %%
