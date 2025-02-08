#%%
def plot_heatmap_and_profiles(array,
    vline_x:int=0, hline_y:int=0, 
    xlabel="X", ylabel="Y", zlabel="Z",
    update_zlim = True,
    x0=0, y0=0, dx=1, dy=1,
    zmin = None,
    zmax = None,
    mask = None,
    cmap = "seismic",
    ):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseButton
    import matplotlib
    #matplotlib.use('Qt5Agg')
    #plt.matplotlib.rcParams['figure.dpi'] = 300

    if mask is not None:
        array = np.ma.array(array, mask = mask)

    X,Y = np.shape(array)
    aspect_ratio = X/Y

    if zmin is None:
        zmin = np.min(array)
    if zmax is None:
        zmax = np.max(array)
    
    if isinstance(zmin, str):
        perc = float(zmin)
        zmin = np.percentile(array, perc)

    if isinstance(zmax, str):
        perc = float(zmax)
        zmax = np.percentile(array, perc)

    extent = [x0, x0+X*dx, y0, y0+Y*dy]
    xlim = (x0, x0+X*dx)
    ylim = (y0, y0+Y*dy)
    x_arr = np.arange(*xlim, dx)
    y_arr = np.arange(*ylim, dy)


    get_slices = lambda: (array[vline_x], array[:,hline_y])
    get_zvalue = lambda: array[vline_x, hline_y]
    if update_zlim:
        get_zlimits = lambda: (max(np.max(vline_data), np.max(hline_data)),
                            min(np.min(vline_data), np.min(hline_data)))
    else:
        get_zlimits = lambda: (zmax, zmin)


    vline_data, hline_data = get_slices()
    slices_zmax, slices_zmin = get_zlimits()
    zvalue = get_zvalue()


    fig = plt.figure(
        #layout='constrained'
        )
    

    if aspect_ratio>=1:
        y_scale = 1
        x_scale = 1/aspect_ratio
        top = 0.5
        right = top/aspect_ratio
    else:
        x_scale = 1
        y_scale = aspect_ratio
        right = 0.5
        top = right*aspect_ratio

    ax = fig.add_gridspec(
        top=1-top, right=1-right,
        #left = 0.1, bottom = 0.1,
        ).subplots()
    
    ax.set(aspect=1)
    ax_x = ax.inset_axes([0, 1.05, 1, y_scale])
    ax_y = ax.inset_axes([1.05, 0, x_scale, 1])
    cbar_ax = ax_y.inset_axes([0.1, 1.15, 0.85, 0.05])
    
    ax_x.set_xlim(*xlim)
    ax_x.set_ylim(slices_zmin, slices_zmax)

    ax_y.set_ylim(*ylim)
    ax_y.set_xlim(slices_zmin, slices_zmax)
    
    ax_x.set_xticks([])   
    ax_y.set_yticks([])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax_x.set_ylabel(zlabel)
    ax_y.set_xlabel(zlabel)

    cmap_ = matplotlib.cm.get_cmap(cmap)
    cmap_.set_bad(color='green')
    im = ax.imshow(
        array.T, 
        cmap=cmap_, 
        interpolation='none', 
        origin = "lower",
        extent = extent,
        vmin = zmin,
        vmax = zmax
        )
    


    
    hline_mark = ax.axhline(y=y_arr[hline_y], color = "black", linewidth = 0.7)
    vline_mark = ax.axvline(x=x_arr[vline_x], color = "black", linewidth = 0.7)

    cbar = fig.colorbar(im, ax=ax, cax = cbar_ax, 
                        orientation = "horizontal",
                        )
    cbar.set_label(zlabel, labelpad = -40)

    vline_profile, = ax_y.plot(vline_data, y_arr, color = "black")
    vline_scatter, = ax_y.plot(x_arr[vline_x], zvalue, color = "black", marker = 's', markerfacecolor="none")
    
    hline_profile, = ax_x.plot(x_arr, hline_data, color = "black")
    hline_scatter, = ax_x.plot(zvalue, y_arr[hline_y], color = "black", marker = 's', markerfacecolor="none")
    
    
    fig_size=7
    if aspect_ratio>=1:
        fig.set_size_inches(fig_size,fig_size/aspect_ratio+0.5)
    else:
        fig.set_size_inches(fig_size*aspect_ratio,fig_size+0.5)

    text = ax_x.text(
        1.05, 0.95, 
        f"{xlabel}={x_arr[vline_x]}\n{ylabel}={y_arr[hline_y]}\n{zlabel}={zvalue:.3f}", 
        transform=ax_x.transAxes, 
        ha="left", 
        va = "top", 
        fontsize=13
        )
    
    def update_plots():
        nonlocal vline_x, hline_y
        nonlocal vline_data, hline_data 
        nonlocal slices_zmax, slices_zmin
        nonlocal zvalue
        nonlocal x_arr, y_arr
        nonlocal dx, dy

        vline_x = int(vline_x%X)
        hline_y = int(hline_y%Y)

        vline_data, hline_data = get_slices()
        slices_zmax, slices_zmin = get_zlimits()
        zvalue = get_zvalue()

        ax_x.set_ylim(slices_zmin, slices_zmax)
        ax_y.set_xlim(slices_zmin, slices_zmax)

        vline_profile.set_xdata(vline_data)
        hline_profile.set_ydata(hline_data)

        hline_mark.set_ydata([y_arr[hline_y]])
        vline_mark.set_xdata([x_arr[vline_x]])

        vline_scatter.set_ydata([y_arr[hline_y]])
        vline_scatter.set_xdata([zvalue])

        hline_scatter.set_ydata([zvalue])
        hline_scatter.set_xdata([x_arr[vline_x]])

        text.set_text(f"{xlabel}={x_arr[vline_x]}\n{ylabel}={y_arr[hline_y]}\n{zlabel}={zvalue:.3f}")

        fig.canvas.draw()

    def on_click(event):
        nonlocal vline_x, hline_y
        nonlocal vline_data, hline_data 
        nonlocal slices_zmax, slices_zmin
        nonlocal zvalue
        nonlocal x_arr, y_arr
        nonlocal dx, dy

        if event.button is MouseButton.LEFT:
            x, y = event.xdata, event.ydata
            vline_x = int((x+x0)/dx)
            hline_y = int((y+y0)/dy)
            
            update_plots()
    
    def on_press(event):
        
        nonlocal vline_x, hline_y
        nonlocal vline_data, hline_data 
        nonlocal slices_zmax, slices_zmin
        nonlocal zvalue
        nonlocal x_arr, y_arr
        nonlocal dx, dy

        if event.key == 'right':
            vline_x=vline_x+1
        elif event.key == 'left':
            vline_x=vline_x-1
        elif event.key == 'down':
            hline_y=hline_y-1
        elif event.key == 'up':
            hline_y=hline_y+1
        elif event.key == "c":
            vline_x = X//2
            hline_y = Y//2
    
        update_plots()

    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id) 
    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.manager.set_window_title(zlabel)
    return fig


#%%
if __name__ == "__main__":
    import numpy as np
    row_values = np.linspace(0, 1, 100)  # shape (100,)
    arr = row_values[:, None] + np.zeros((1, 100))
    #%matplotlib qt
    plot_heatmap_and_profiles(arr, vline_x=50, hline_y=50)
# %%
