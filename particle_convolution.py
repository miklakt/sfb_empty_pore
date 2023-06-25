#%%
from generate_spherical_kernel import generate_spherical_kernel
import matplotlib.pyplot as plt
import pandas as pd
import utils
import numpy as np
from scipy.signal import convolve
# %%
def convolve_particle_volume(array, particle_d):
    nr, nz = np.shape(array)
    particle_r = int(particle_d/2)
    array = np.pad(array, ((0, particle_r),(particle_r,particle_r-1)), 'edge')
    new_arr = []
    for r in range(nr):
        volume, surface, extent = generate_spherical_kernel(particle_r,r)

        zconv_result=convolve(array[int(extent[0]):int(extent[1])], volume, mode="valid")#[:,1:]#, method = "direct")

        #conv_input_arr = array[int(extent[0]):int(extent[1])]
        #zconv_result = [np.sum(conv_input_arr[:,z-particle_r:z+particle_r] * volume) for z in range(particle_r, particle_r+nz)]


        new_arr.append(zconv_result)
    new_arr = np.vstack(new_arr)
    #new_arr = (new_arr[:,1:]+new_arr[:,:-1])/2
    return new_arr

def convolve_particle_surface(array, particle_d):
    nr, nz = np.shape(array)
    particle_r = int(particle_d/2)
    array = np.pad(array, ((0, particle_r),(particle_r,particle_r-1)), 'edge')
    new_arr = []
    for r in range(nr):
        volume, surface, extent = generate_spherical_kernel(particle_r,r)

        zconv_result=convolve(array[int(extent[0]):int(extent[1])], surface, mode = 'valid')#[:,1:]

        #conv_input_arr = array[int(extent[0]):int(extent[1])]
        #zconv_result = [np.sum(conv_input_arr[:,z-particle_r:z+particle_r] * surface) for z in range(particle_r, particle_r+nz)]

        new_arr.append(zconv_result)
    new_arr = np.vstack(new_arr)
    #new_arr = (new_arr[:,1:]+new_arr[:,:-1])/2
    return new_arr

def cylynder_r0_kernel(radius:int, height:int = None):
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
volume, surface = cylynder_r0_kernel(4,8)

#%%
#%%
if __name__ == "__main__":
    df = pd.read_pickle("empty_brush.pkl")
    phi = utils.get_by_kwargs(df, chi_PS = 1, r=26, s = 52)["phi"].squeeze()
    phi_conv = convolve_particle_surface(phi, 16)
    def mirror(arr):
        cut = arr
        return np.vstack((np.flip(cut), cut[:,::-1])).T
    plt.imshow(mirror(phi_conv))
    #plt.imshow(phi)
# %%
