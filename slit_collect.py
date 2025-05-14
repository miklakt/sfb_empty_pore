#%%
import sfbox_utils
import numpy as np
import matplotlib.pyplot as plt
import extract_features
import pandas as pd
import pathlib
#%%
dir = pathlib.Path("temp12")
#%%
def process_data(raw_data):
    import extract_features
    data = extract_features.slit_geometry(raw_data)
    data.update(extract_features.polymer_density(raw_data, **data))
    data.update(extract_features.polymer_potential(raw_data, **data))
    data.update(extract_features.strands_density(raw_data, **data))
    data.update(extract_features.chi_params(raw_data))
    data.update(extract_features.free_energy(raw_data))
    data["sigma"] = round(data["sigma"], 4)
    data["comment"] = "increasing_chi_PS"
    return data

def name_file(data, timestamp = True):
    keys = ["chi_PS", "r", "xlayers", "ylayers", "N", "sigma"]
    name = "_".join([f"{k}_{data[k]}" for k in keys])
    if data.get("comment"):
        name = "_".join([data['comment'], name])
    return name
# %%
for filename in dir.glob("*.out"):
    print(filename)
    sfbox_utils.store.store_file_sequential(
        file=filename,
        process_routine=process_data,
        naming_routine=name_file,
        dir = "h5_slit",
        on_file_exist="keep",
    )
# %%
master = sfbox_utils.store.create_reference_table(storage_dir="h5_slit")
# %%
master.to_pickle("pkl/reference_table_slit.pkl")
# %%
%matplotlib qt
from utils import get_by_kwargs
from heatmap_explorer import plot_heatmap_and_profiles
import scf_pb
from planar_poor_pore_analytic.symbolic_convex_planar import Y as Y_convex
from planar_poor_pore_analytic.symbolic_convex_planar import h_V
from matplotlib.patches import Rectangle

#N=300
#sigma = 0.04
#chi_PS = 1.2
#chi_PW = -1
#r=40
#s=40

#data = get_by_kwargs(master, r=r, N = N, sigma = sigma, chi_PS = chi_PS, chi_PW = chi_PW)
#%%
data = get_by_kwargs(master, r=20, N = "largest", chi_PS = 0.8)
N = data.N.squeeze()
sigma = data.sigma.squeeze()
chi_PS = data.chi_PS.squeeze()
chi_PW = data.chi_PW.squeeze()
dataset = data.dataset["phi"].squeeze()

l1 = data.l1.squeeze()
s = data.s.squeeze()
h = data.h.squeeze()
ylayers = data.ylayers.squeeze()
r = data.r.squeeze()

H = scf_pb.D(N=N, sigma = sigma, chi = chi_PS)
z = np.linspace(0, H)
phi_av = np.sum(scf_pb.phi_v(N=N, sigma=sigma, chi=chi_PS, z=z))/len(z)
theta = N*sigma*s
volume = theta/phi_av

h = h_V(s/2, volume/2)-1e-3
x = np.linspace(0,h)
y = np.array([Y_convex(s/2, volume/2, r)(x_) for x_ in x])

fig = plot_heatmap_and_profiles(dataset)
ax = fig.axes[0]
ax.plot(r-x, y+l1+s/2, color = "yellow")
ax.plot(r-x, -y+l1+s/2, color = "yellow")

ax.add_patch(Rectangle((r,l1),r,s, color = "grey", edgecolor = "black")) 
fig
# %%
from planar_poor_pore_analytic.symbolic_convex_planar import total_surface_energy
total_surface_energy(chi_PS, -1, s-2, volume, r)
# %%
