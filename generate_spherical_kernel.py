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
    kernel = generate_sphere_volume_surface_kernel(10, 57)
    plt.pcolormesh(kernel[0], edgecolors='k', linewidth=2)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    #plt.savefig("vol.svg")
# %%
