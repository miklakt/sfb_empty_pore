#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
#%%
df = pd.read_pickle("data/empty_brush.pkl_old")
# %%
df = df.loc[(df["xlayers"]==60)&(df["chi_PW"]==0.)]
df.phi = df.apply(lambda _: np.resize(_.phi, (_.xlayers,_.ylayers)),axis=1)
#%%
fig, ax = plt.subplots()
for idx, grouped in df.groupby(by = "chi_PS"):
    phi = grouped.phi.squeeze()
    xlayers = grouped.xlayers.squeeze()
    phi_x0=phi[0]
    z = list(range(len(phi_x0)))
    f = np.vectorize(interp1d(z, phi_x0))
    ax.plot(z, f(z), label = idx)
    #ax.scatter(z, phi_x0)
ax.legend(title = "$\chi_{PS}$")
ax.set_xlabel("z")
ax.set_ylabel("$\phi$")
plt.savefig("pore_central_axis_phi.pdf")
# %%
coefs = [0.16, 0.08]

def free_energy(phi , a1, a2, chi_PC, chi_PS, d):
    Pi = -np.log(1-phi)-chi_PS*phi**2-phi
    V = np.pi*d**3/4
    S = 3/2*np.pi*d
    F_V = Pi*V
    chi_ads = chi_PC - chi_PS*(1-phi)
    chi_crit = 6*np.log(5/6)
    gamma = (chi_ads - chi_crit)*coefs[0]*phi+ coefs[1]*phi**2
    F_S = gamma*S
    return F_V+F_S
free_energy_func = np.vectorize(free_energy, excluded=["a1", "a2" , "chi_PC", "chi_PS", "d"])

#%%
from diffusion import D_eff
# %%
chi_pc = -1.5
d=10
fig, ax = plt.subplots()
for idx, grouped in df.groupby(by = "chi_PS"):
    U = grouped.apply(lambda _: free_energy_func(_.phi[0], *coefs, chi_pc, _.chi_PS, d),axis=1).squeeze()
    ax.plot(U, label = idx)
ax.legend(title = "$\chi_{PS}$")
ax.set_xlabel("z")
ax.set_ylabel("$F/k_BT$")
plt.savefig("pore_central_fe.pdf")
# %%
chi_PCs = np.linspace(0.5, -2.5, 20)
d=10
fig, ax = plt.subplots()
ax.set_xlabel("$\chi_{PC}$")
ax.set_ylabel("$D_{eff}$")
for idx, grouped in df.groupby(by = "chi_PS"):
    D = [] 
    for chi_pc in chi_PCs:
        U = grouped.apply(lambda _: free_energy_func(_.phi[0], *coefs, chi_pc, _.chi_PS, d),axis=1).squeeze()
        D.append(D_eff(b = len(U)-1, U_x = U))
    ax.scatter(chi_PCs, D, label = idx)
ax.legend(title = "$\chi_{PS}$")
plt.savefig("pore_D_eff_on_chi_PC.pdf")
# %%
chi_PCs = [0.5, -0.0, -0.75, -1.5, -2.5]
d=10
fig, ax = plt.subplots()
ax.set_xlabel("$\chi_{PS}$")
ax.set_ylabel("$D_{eff}$")
for chi_pc in chi_PCs:
    D = []
    chi_ps = []
    for idx, grouped in df.groupby(by = "chi_PS"):
        U = grouped.apply(lambda _: free_energy_func(_.phi[0], *coefs, chi_pc, _.chi_PS, d),axis=1).squeeze()
        D.append(D_eff(b = len(U)-1, U_x = U))
        chi_ps.append(idx)
    ax.scatter(chi_ps, D, label = chi_pc)
ax.legend(title = "$\chi_{PC}$")
plt.savefig("pore_D_eff_on_chi_PS.pdf")
# %%
ax.set_yscale("log")
#%%
plt.savefig("pore_D_eff_on_chi_PS_semilog.pdf")
# %%
