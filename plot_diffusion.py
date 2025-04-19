#%%
import numpy as np
from calculate_fields_in_pore import    mobility_Rubinstein, \
                                        mobility_Phillies#, \
                                        #mobility_FoxFlory, \
                                        #mobility_Hoyst

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

def rubinstein_diffusion(phi, d, nu):
    """
    Computes the diffusion coefficient using the Rubinstein scaling model with a nu-dependent equation.

    Parameters:
    D0 : float
        Diffusion coefficient in pure solvent (m^2/s)
    r : float
        Particle radius (m)
    c : float
        Polymer concentration (g/mL)
    a : float
        Monomer size (m)
    nu : float
        Scaling exponent (depends on Flory-Huggins parameter χ)
    
    Returns:
    D : float
        Effective diffusion coefficient in polymer solution (m^2/s)
    """
    # Compute mesh size (xi) using scaling relation
    #xi = np.where(phi==0, 0.0, (phi ** -nu))
    xi = np.where(phi==0, 0.0, (phi ** (-nu/(3*nu-1))))
    
    # Compute scaling exponent m
    m = 2 / nu
    
    # Compute effective diffusion coefficient using Rubinstein model
    D = np.where(xi==0, 1.0, (d / xi) ** -m)
    k=1
    D = D /(1.0 + D**k)**(1 / k)
    
    return D

def mobility_Rubinstein_3(phi, d, N):
    D = np.where(phi==0, 1.0, N**(1/2)*phi**(-3)/d*np.exp(d**2/N))
    k=1
    D = D /(1.0 + D**k)**(1 / k)
    return D

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
#%%
phi = np.geomspace(0.001, 0.7)
nu = [0.588, 0.5]
d = 8

D = [rubinstein_diffusion(phi, d, nu_ ) for nu_ in nu]

fig, ax = plt.subplots()
for nu_, D_ in zip(nu, D):
    ax.plot(phi, D_, label = nu_)

ax.set_xscale("log")
ax.set_xlabel("$\phi$")

ax.set_yscale("log")
ax.set_ylabel("$D/D_0$")

ax.legend(title = r"$\nu$")

# %%
fig, ax = plt.subplots()
phi = np.geomspace(0.1, 0.8)
d = 8

#D_Hoyst = mobility_Hoyst(phi, d = d, alpha=40, delta = 0.7)
D_Hoyst = mobility_Hoyst(phi, N=300, d = d, alpha=1.63, delta = 0.89)
D_Rubinstein = mobility_Rubinstein(phi, k=1, d=d, prefactor=1)
#D_Rubinstein2 = rubinstein_diffusion(phi, d=d, nu = 0.5)
#D_Rubinstein3 = mobility_Rubinstein_3(phi, d, N=300)
D_Phillies = mobility_Phillies(phi, beta = 8, nu = 0.76)
#D_FoxFlory = mobility_FoxFlory(phi, N=300)

ax.plot(phi, D_Rubinstein, label = 'Rubinstein')
#ax.plot(phi, D_Rubinstein2, label = 'Rubinstein2')
#ax.plot(phi, D_Rubinstein3, label = 'Rubinstein3')
# ax.plot(phi, D_FoxFlory, label = 'Fox-Flory')
ax.plot(phi, D_Hoyst, label = 'Hoyst(empirical)')
ax.plot(phi, D_Phillies, label = 'Phillies(empirical)')

#ax.set_xscale("log")
ax.set_xlabel("$\phi$")

ax.set_yscale("log")
ax.set_ylabel("$D/D_0$")

ax.legend(title = "Polymer diffusion model")
# %%
