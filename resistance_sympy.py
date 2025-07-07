#%%
import sympy as sp
D, r_p, z, z_b = sp.symbols("D r_p z z_b", positive  = True)
#%%
rho_ext = D*2*sp.pi*(z**2+r_p**2)
# %%
R_ext = sp.integrate(rho_ext**-1, (z, 0, "+oo"))
# %%
R_to_boundary = sp.integrate(rho_ext**-1, (z, z_b, "+oo"))
# %%
rho_cyl = D*(2*sp.pi*(r_p+z)*z + sp.pi*(r_p+z)**2)
# %%
# R_layer_cyl = sp.integrate(rho_cyl**-1, (z, z_b, z_b-1))
R_layer_cyl= 2*sp.pi*D*((r_p+z)**2 + sp.log((r_p+z)/(r_p+z-1))*z)
R_layer_cyl = R_layer_cyl**-1
# %%
R_layer = sp.integrate(rho_ext**-1, (z, z_b-1, z_b))
# %%
f = (R_layer/R_layer_cyl).simplify()
# %%
