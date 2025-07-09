#%%
from scf_pb import generate_sphere_volume_surface_kernel
from joblib import Memory
import matplotlib.pyplot as plt
memory = Memory("__func_cache__", verbose=1)
#%%
#@pickle_cache.pickle_lru_cache()
@memory.cache
def generate_spherical_kernel(radius, r):
    print(f"calculating the kernel for {radius=}, {r=}")
    volume, surface, extent = generate_sphere_volume_surface_kernel(radius, r)
    extent = (extent[0], extent[1])
    return volume, surface, extent

# #%%
# if __name__ == "__main__":
#     for radius in range(2,10):
#         for r in range(70):
#             kernel = generate_spherical_kernel(radius, r)
# %%
# if __name__ == "__main__":
#     kernel = generate_sphere_volume_surface_kernel(10, 12)
#     plt.pcolormesh(kernel[0], edgecolors='k', linewidth=2)
#     ax = plt.gca()
#     ax.set_aspect('equal')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     #plt.savefig("vol.svg")
# %%
# import numpy as np
# def dist(r, theta, z, r_c, z_c):
#     return np.sqrt(r**2 + r_c**2 - 2*r*r_c*np.cos(theta) + (z-z_c)**2)

# def delta_dist(d, r, theta, z, r_c, z_c, tol = 2e-2):
#     return abs(d/2-dist(r, theta, z, r_c, z_c))<tol

# def H_dist(d, r, theta, z, r_c, z_c):
#     return dist(r, theta, z, r_c, z_c)-d/2<0

# def dA(d, r, theta, z, r_c, z_c):
#     arr = delta_dist(d, r, theta, z, r_c, z_c)*dist(r, theta, z, r_c, z_c)*r
#     arr2 = (r_c*np.abs(np.sin(theta)))
#     arr = np.where(arr2==0, 0, arr/arr2)
#     return arr


#%%
# d = 8
# r_c = 2
# z_c = 0
# dr = 0.01
# dz = 0.01
# dtheta = 0.005
# r=np.arange(max(r_c - d/2, 0), r_c+d/2, dr)
# z=np.arange(z_c-d/2, z_c+d/2,dz)
# theta = np.arange(0, 2*np.pi, dtheta)

# V = [[np.sum(H_dist(d, r_, theta, z_, r_c, z_c))*dtheta*r_ for z_ in z] for r_ in r]
# # Meshgrids
# # R, Z, Theta = np.meshgrid(r, z, theta, indexing='ij')  # shape: (len(r), len(z), len(theta))
# # mask = H_dist(d, R, Theta, Z, r_c, z_c)
# # V_theta = np.sum(mask, axis=2) * dtheta * r[:, None]  # shape: (len(r), len(z))
# # V = V_theta.tolist()
# #%%
# S = [[np.sum(dA(d, r_, theta, z_, r_c, z_c))*dtheta for z_ in z] for r_ in r]
# #S[S == np.inf] = 0
# #%%
# # voxel = ((H+r)*H)
# # V = np.sum(((H+r)*H),axis=0).T*dtheta
# plt.imshow(V, origin = "lower", extent = [min(z), max(z), min(r), max(r)], vmin = 0, vmax = 20)
# #%%
# print("Volume formula", np.pi*d**3/6)
# print("Volume discrete",np.sum(V)*dr*dz)
# # %%
# plt.imshow(S, origin = "lower", extent = [min(z), max(z), min(r), max(r)], vmin = 0, vmax = 20)
# # %%
# print("Surface formula", np.pi*d**2)
# print("Surface discrete", np.sum(S)*dr*dz)
# %%
