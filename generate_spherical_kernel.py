#%%
from scf_pb import generate_sphere_volume_surface_kernel
import pandas as pd
#%%
pkl_file = "spherical_kernels.pkl"
#%%
def generate_spherical_kernel(radius, r):
    df = pd.read_pickle(pkl_file)
    stored = df.query(f"radius == {radius} & r=={r}")
    if len(stored) == 0:
        print(f"calculating the kernel for {radius=}, {r=}")
        volume, surface, extent = generate_sphere_volume_surface_kernel(radius, r)
        extent = (extent[0], extent[1])
        row = {
            "radius":radius,
            "r": r,
            "volume": volume,
            "surface": surface,
            "extent": extent
            }
        df = df.append(row, ignore_index = True)
        df.to_pickle(pkl_file)
    else:
        stored = dict(stored.squeeze())
        volume = stored["volume"]
        surface = stored["surface"]
        extent = stored["extent"]
    return volume, surface, extent

#%%
if __name__ == "__main__":
    for radius in range(16,17):
        for r in range(70):
            generate_spherical_kernel(radius, r)
# %%
