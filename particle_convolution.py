#%%
from generate_spherical_kernel import generate_spherical_kernel
import matplotlib.pyplot as plt
import pandas as pd
import utils
import numpy as np
from scipy.signal import convolve
# %%
def convolve_particle_volume(array, particle_d, mode = "valid"):
    nr, nz = np.shape(array)
    particle_r = int(particle_d/2)
    array = np.pad(array, ((0, particle_r),(particle_r,particle_r)), 'edge')
    new_arr = []
    for r in range(nr):
        volume, surface, extent = generate_spherical_kernel(particle_r,r)
        zconv_result=convolve(array[int(extent[0]):int(extent[1])], volume, mode="valid")
        new_arr.append(zconv_result)
    new_arr = np.vstack(new_arr)
    if mode == "valid":
        new_arr = new_arr[:,:-1]
    elif mode == "same":
        new_arr = (new_arr[:,1:]+new_arr[:,:-1])/2
    else:
        raise ValueError("Invalid convolution mode")
    return new_arr

def convolve_particle_surface(array, particle_d, mode = "valid"):
    nr, nz = np.shape(array)
    particle_r = int(particle_d/2)
    array = np.pad(array, ((0, particle_r),(particle_r,particle_r)), 'edge')
    new_arr = []
    for r in range(nr):
        volume, surface, extent = generate_spherical_kernel(particle_r,r)
        zconv_result=convolve(array[int(extent[0]):int(extent[1])], surface, mode = 'valid')#[:,1:]
        new_arr.append(zconv_result)
    
    new_arr = np.vstack(new_arr)
    if mode == "valid":
        new_arr = new_arr[:,:-1]
    elif mode == "same":
        new_arr = (new_arr[:,1:]+new_arr[:,:-1])/2
    else:
        raise ValueError("Invalid convolution mode")
    return new_arr

def cylinder_r0_kernel(radius:int, height:int = None):
    if height is None:
        height = radius*2
    r = np.arange(radius)+1/2
    volume_r = np.pi*(2*r+1)
    volume = np.tile(volume_r, (height,1)).T
    surface = np.zeros_like(volume)

    surface[-1,:] = 2*np.pi*radius
    surface[:, 0] = surface[:,-1] = volume_r

    return volume, surface
#%%
#%%
if __name__ == "__main__":
    import calculate_fields_in_pore
    df = pd.read_pickle("pkl/reference_table_empty_brush.pkl")
    phi = utils.get_by_kwargs(df, chi_PS = 1, r=26, s = 52).dataset["phi"].squeeze()
    a0, a1 = 0.70585835, -0.31406453
    gamma = calculate_fields_in_pore.gamma(0.1, -2, phi, a0, a1)
    phi_conv = convolve_particle_volume(phi, 20, "valid")
    def mirror(arr):
        cut = arr
        return np.vstack((np.flip(cut), cut)).T
    plt.imshow(mirror(phi_conv))
    #plt.imshow(phi)
# %%
    volume, surface, extent = generate_spherical_kernel(10,57)
    zconv_result=convolve(phi[int(extent[0]):int(extent[1])], volume, mode="valid")
    plt.imshow(zconv_result)
# %%
