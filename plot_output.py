#%%
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import pathlib
#%%
#r = pickle.load(open("data/raw/f92d0394-4e1c-46b4-9818-209fbf709f41.pkl", "rb"))

for file in pathlib.Path("data/raw").glob("*.pkl"):
    r = pickle.load(open(file, "rb"))
    s=40
    xlayers = r["lat:2G:n_layers_x"]
    ylayers = r["lat:2G:n_layers_y"]
    composition = r["mol:pol0:composition"]
    N = int(composition.rsplit(")")[-1])+1
    chi_PS = r["chi list:P0:chi - S"]
    chi_PW = r["chi list:P0:chi - W"]
    phi = r["mon:P:phi:profile"].reshape((xlayers,ylayers))
    #phi_0 = r["mon:P1:phi:profile"].reshape((xlayers,ylayers))
    phi_0=np.sum([r[f"mon:P{i}:phi:profile"].reshape((xlayers,ylayers)) for i in range(s)], axis =0)
    phi = phi+phi_0
    phi_m = np.vstack([np.flip(phi, axis = 0),phi])
    fig, ax = plt.subplots()
    c = ax.imshow(phi_m, origin = "lower", aspect = "equal", cmap = "RdBu_r", vmin =0 , vmax = 0.5)
    ax.contour(phi_m, colors = 'black', alpha = 1)
    ax.set_aspect(aspect = 1)
    plt.title("$\chi_{PS}=$"+f"{chi_PS}" + f"$, N={N}, \sigma = 0.02$")
    plt.colorbar(c)
    plt.show()
# %%
fig = go.Figure(data =
    [
        go.Contour(
        z=phi,
        line_smoothing=0
    )]
    )
fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )
fig.show('browser')
# %%
fig, ax = plt.subplots()
ax.plot(range(xlayers), phi[:,int(ylayers/2)], label = 0)
ax.plot(range(xlayers), phi[:,int(ylayers/2)-10], label = 10)
ax.plot(range(xlayers), phi[:,int(ylayers/2)-20], label = 20)
ax.plot(range(xlayers), phi[:,int(ylayers/2)-30], label = 30)
ax.legend(title = "z")
plt.savefig("fig.pdf")
# %%
