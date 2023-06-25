#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import get_by_kwargs

#matplotlib settings
LAST_USED_COLOR = lambda: plt.gca().lines[-1].get_color()
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

def plot_imshow(
    dataframe, 
    array_name, 
    contour = True, 
    imshow_kwargs = {}, 
    patch = True, 
    patches_kwargs = {}, 
    colorbar = True,  
    **kwargs
    ):

    imshow_kwargs_default = dict(origin = "lower", aspect = "equal", cmap = "RdBu_r")
    imshow_kwargs_default.update(imshow_kwargs)

    print(kwargs)

    plot_data = get_by_kwargs(dataframe, **kwargs).squeeze()
    X = plot_data[array_name].reshape((plot_data["xlayers"],plot_data["ylayers"]))
    X_m = np.vstack([np.flip(X, axis = 0),X])
    fig, ax = plt.subplots()

    c = ax.imshow(X_m, **imshow_kwargs_default)
    if contour:
        ax.contour(
            X_m,
            colors = 'black',
            alpha = 1,
            linestyles = 'solid',
            linewidths = 0.4,
        )
    ax.set_aspect(aspect = 1)
    ax.set_title(" ".join([str(k)+"="+str(v) for k,v in kwargs.items()]))

    if colorbar:
        fig.colorbar(c)

    if patch:
        rect = patches.Rectangle((plot_data.l1-1, 0), plot_data.s, plot_data.h-1, **patches_kwargs)
        ax.add_patch(rect)

        rect = patches.Rectangle((plot_data.l1-1, plot_data.xlayers*2-plot_data.h), plot_data.s, plot_data.h-1, **patches_kwargs)
        ax.add_patch(rect)

    return fig

def plot_imshow_dict(
    data, 
    array_name, 
    contour = True, 
    imshow_kwargs = {}, 
    patch = True, 
    patches_kwargs = {}, 
    colorbar = True,  
    **kwargs
    ):
    imshow_kwargs_default = dict(origin = "lower", aspect = "equal", cmap = "RdBu_r")
    imshow_kwargs_default.update(imshow_kwargs)
    
    xlayers = data["xlayers"]
    ylayers = data["ylayers"] 

    X = data[array_name].reshape((xlayers,ylayers))
    X_m = np.vstack([np.flip(X, axis = 0),X])
    fig, ax = plt.subplots()

    c = ax.imshow(X_m, **imshow_kwargs_default)
    if contour:
        ax.contour(
            X_m,
            colors = 'black',
            alpha = 1,
            linestyles = 'solid',
            linewidths = 0.4,
        )
    ax.set_aspect(aspect = 1)
    ax.set_title(" ".join([str(k)+"="+str(v) for k,v in kwargs.items()]))

    if colorbar:
        fig.colorbar(c)

    if patch:
        l1 = data["l1"]
        s = data["s"]
        h = data["h"]
        rect = patches.Rectangle((l1-1, 0), s, h, **patches_kwargs)
        ax.add_patch(rect)

        rect = patches.Rectangle((l1-1, xlayers*2-h), s, h-1, **patches_kwargs)
        ax.add_patch(rect)

    plt.show()
    return fig

#%%