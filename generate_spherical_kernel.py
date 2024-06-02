#%%
from scf_pb import generate_sphere_volume_surface_kernel
from joblib import Memory
memory = Memory("__func_cache__", verbose=1)
#%%
#@pickle_cache.pickle_lru_cache()
@memory.cache
def generate_spherical_kernel(radius, r):
    print(f"calculating the kernel for {radius=}, {r=}")
    volume, surface, extent = generate_sphere_volume_surface_kernel(radius, r)
    extent = (extent[0], extent[1])
    return volume, surface, extent

#%%
if __name__ == "__main__":
    for radius in range(2,10):
        for r in range(70):
            kernel = generate_spherical_kernel(radius, r)
# %%
if __name__ == "__main__":
    kernel = generate_sphere_volume_surface_kernel(10, 12)
    plt.pcolormesh(kernel[0], edgecolors='k', linewidth=2)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    #plt.savefig("vol.svg")
# %%
import numpy as np
def dist(r, theta, z, r_c, z_c):
    return np.sqrt(r**2 + r_c**2 - 2*r*r_c*np.cos(theta) + (z-z_c)**2)

def delta_dist(d, r, theta, z, r_c, z_c):
    return abs(d/2-dist(r, theta, z, r_c, z_c))<5e-3

def H_dist(d, r, theta, z, r_c, z_c):
    return dist(r, theta, z, r_c, z_c)-d/2<0

# %%
dr = 0.02
dz = 0.02
r=np.arange(0, 4,dr)
z=np.arange(0, 4,dz)
dtheta = 0.1
theta = np.arange(0, 2*np.pi, dtheta)
r_c = 1
z_c = 2
d = 3
#delta = np.array([[delta_dist(d, r_, 0.0, z_, r_c, z_c) for r_ in r] for z_ in z])
H = np.array([[[H_dist(d, r_, theta_, z_, r_c, z_c) for r_ in r] for z_ in z] for theta_ in theta])
#V =
# %%
voxel = ((H+r)*H)
V = np.sum(((H+r)*H),axis=0).T
plt.imshow(V, origin = "lower", extent = [min(z), max(z), min(r), max(r)])
# %%
np.pi*d**3/6
# %%
# ax = plt.figure().add_subplot(projection='3d')
# ax.voxels(voxel)
# %%
np.sum(voxel)*dr*dz*dtheta/2
# %%
