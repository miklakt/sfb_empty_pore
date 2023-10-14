#%%
import tqdm
import matplotlib.pyplot as plt
import matplotlib.transforms
import tqdm
import pickle
import numpy as np
import h5py
from heatmap_explorer import plot_heatmap_and_profiles

def get_flux_empty_pore_theory(D, r, c, L):
    return 2*D*r*c/(np.pi + 2*L/r)

def shrink_cbar(ax, shrink=0.9):
    b = ax.get_position()
    new_h = b.height*shrink
    pad = (b.height-new_h)/2.
    new_y0 = b.y0 + pad
    new_y1 = b.y1 - pad
    b.y0 = new_y0
    b.y1 = new_y1
    ax.set_position(b)

#%%
pore_radius = 26 # pore radius
wall_thickness = 52 # wall thickness
d = 24
chi_PC = -1.25
chi = 0.5
walls_only = False
dt = 0.1
zlayers = 392+100
rlayers = 66+200
differencing = "power_law"

flux_empty_pore_theory = get_flux_empty_pore_theory(
    D = 1, r = pore_radius-d/2, c=1, L = wall_thickness
)

if walls_only:
    simulation_name = \
    f"simulation_data/{d=}_{zlayers=}_{rlayers=}_{dt=}_{differencing}.h5"
else:
    simulation_name = \
    f"simulation_data/{d=}_{chi=}_{chi_PC=}_{zlayers=}_{rlayers=}_{dt=}_{differencing}.h5"

#simulation_name = "simulation_data/d=8_chi=0.4_chi_PC=-1_zlayers=492_rlayers=216_every=10000_dt=0.01"
# %%
def plot_all(dd_obj):
    from mpl_toolkits.axes_grid1 import ImageGrid
    from matplotlib.gridspec import GridSpec
    fig  = plt.figure(
            figsize = (6,7.5),
            dpi=600,
            #constrained_layout = True
        )
    
    def plot_func(i=-1):
        if i<0:
            i = len(dd_obj["timestep"])+i
        fig.clear()
        fig.set_size_inches(5, 8.5)
        gs = GridSpec(
            #4, 2,
            5, 2, 
            figure = fig, 
            height_ratios=[
                1, 
                1, 
                1.5, 
                1.5, 
                1.5,
                ], 
            width_ratios =[1,0.05],
            hspace = 0.1
            )
        gs.update(left=0.2)

        main_ax = fig.add_subplot(gs[2,0])
        axd = {
            "J_z" : fig.add_subplot(gs[0,0], sharex = main_ax),
            "R_z" : fig.add_subplot(gs[1,0], sharex = main_ax),
            #"int_R_z" : fig.add_subplot(gs[2,0], sharex = main_ax),
            "c_arr" : main_ax,
            "J_arr" : fig.add_subplot(gs[3,0]),
            "div_J" : fig.add_subplot(gs[4,0]),
            "c_arr_cbar" : fig.add_subplot(gs[2,1]),
            "J_arr_cbar" : fig.add_subplot(gs[3,1]),
            "div_J_cbar" : fig.add_subplot(gs[4,1]),
        }
        twinx = axd["J_z"].twinx()
        
        axd["J_z"].set_xticklabels([])
        axd["J_z"].set_ylim(0, np.max(dd_obj["J_z_tot"][i])*1.05)
        axd["J_z"].set_ylabel(r"$\int^{r_{pore}} j_{z}(r,z) dr$")
        axd["J_z"].text(0.05, 0.05, f"$<J> = {np.mean(dd_obj['J_z_tot'][i]):.3f}$",
                        ha = "left", va = "bottom")

        twinx.set_ylabel(r"$\frac{1}{r_{pore}^2} \int^{r_{pore}} c(r,z) r dr$")
        
        
        axd["R_z"].set_xticklabels([])
        axd["R_z"].set_ylabel(r"$R_z$")

        axd["J_z"].plot(dd_obj["J_z_tot"][i])
        
        twinx.plot([],[])#DUMMY
        twinx.plot(dd_obj["c_z_average"][i])
        twinx.yaxis.label.set_color(twinx.lines[-1].get_color())

        axd["R_z"].plot(dd_obj["R_z"][i])
        trans = matplotlib.transforms.blended_transform_factory(
                        axd["R_z"].transAxes, axd["R_z"].transData)
        axd["R_z"].text(
            x = 1, 
            y = dd_obj["R_z"][i][-1], 
            s = f" {dd_obj['R_z'][i][-1]:.3f}", 
            transform = trans,
            va = "center",
            ha = 'left'
            )
        axd["R_z"].axhline(
            dd_obj["R_z"][i][-1], 
            color = axd["R_z"].lines[-1].get_color(),
            linewidth = 0.3, linestyle = "--")
        axd["R_z"].axhline(
            0, 
            color = axd["R_z"].lines[-1].get_color(),
            linewidth = 0.3, linestyle = "--")
        
        axd["R_z"].text(
            0.99, 0.01, 
            f"$R_0 = {1/flux_empty_pore_theory:.3f}$", 
            ha = 'right', 
            va = 'bottom', 
            transform = axd["R_z"].transAxes
            )
        

        c_arr_im = axd["c_arr"].imshow(
            dd_obj["c_arr"][i].T, 
            origin = "lower", 
            #cmap = "gnuplot",
            cmap="seismic",
            )
        
        contour = axd["c_arr"].contour(
            dd_obj["c_arr"][i].T, 
            origin = "lower", 
            colors = "black", 
            linewidths = 0.4
            )
        
        contour0 = axd["c_arr"].contour(
            dd_obj["c_arr"][i].T, 
            origin = "lower", 
            colors = "green", 
            levels = [1e-6]
            )
        
        contour1 = axd["c_arr"].contour(
            dd_obj["c_arr"][i].T, 
            origin = "lower", 
            colors = "orange", 
            levels = [np.max(dd_obj["c_arr"][i])*0.999]
            )
        
        
        J_arr_im = axd["J_arr"].imshow(
            np.linalg.norm(dd_obj["J_arr"][i], axis = -1).T, 
            origin = "lower", 
            cmap = "gnuplot",
            )
        
        x = np.arange(0, dd_obj.attrs["zlayers"], 10)
        y = np.arange(0, dd_obj.attrs["rlayers"], 10)
        xx, yy = np.meshgrid(x, y)
        uv = [dd_obj["J_arr"][i][xx_, yy_] for xx_, yy_ in zip(xx, yy)]
        u = np.moveaxis(uv, -1, 0)[0]
        v = np.moveaxis(uv, -1, 0)[1]
        norm = np.linalg.norm(np.array((u, v)), axis=0)
        
        
        #axd["J_arr"].quiver(
        #    xx, yy, u/norm*2, v/norm*2, 
        #    width = 0.003,
            #headlength = 3,

        #    color = 'grey'
        #    )
        
        #start_points_y = np.arange(1, 27,2)
        #start_points_x = np.ones_like(start_points_y)*dd_obj["zlayers"]/2
        start_points_y = np.arange(0, dd_obj.attrs["rlayers"]-1,10)
        start_points_x = np.ones_like(start_points_y)#*dd_obj.zlayers/2

        start_points = np.array([start_points_x, start_points_y]).T
        J_arr_stream = axd["J_arr"].streamplot(
            xx, yy, u, v, 
            #color = norm,
            color = "grey",
            start_points = start_points,
            broken_streamlines = False,
            arrowsize = 0,
            linewidth = 0.3
            )
        

        c_arr_cbar = plt.colorbar(
            c_arr_im,
            cax=axd["c_arr_cbar"],
            #shrink = 0.6 
            )
        c_arr_cbar.ax.set_title("$c$")
        shrink_cbar(c_arr_cbar.ax, 0.8)
        c_arr_cbar.add_lines(contour)
        c_arr_cbar.ax.axhline(0, color = "green")

        J_arr_cbar = plt.colorbar(
            #J_arr_stream.lines,
            J_arr_im,
            cax=axd["J_arr_cbar"],
            #shrink = 0.6
            )
        J_arr_cbar.ax.set_title("$|j|$")
        shrink_cbar(J_arr_cbar.ax, 0.8)

        try:
            div_J_im = axd["div_J"].imshow(
                dd_obj["div_J_arr"][i].T, 
                origin = "lower", 
                cmap = "gnuplot",
                )
            
            div_J_cbar = plt.colorbar(
                #J_arr_stream.lines,
                div_J_im,
                cax=axd["div_J_cbar"],
                #shrink = 0.6
                )
            div_J_cbar.ax.set_title(r"$\nabla \cdot j$")
            shrink_cbar(J_arr_cbar.ax, 0.8)
            axd["J_arr"].set_xticklabels([])
        except KeyError:
            pass

        timestep = int(dd_obj["timestep"][i] * dd_obj.attrs["dt"])
        c_total = float(np.sum(dd_obj["c_z_average"][i]))
        fig.text(0.3, 0, "t: {:>5}; $c_{{total}}$: {:2f}".format(timestep, c_total), ha = "left")
        
    return fig, plot_func
# %%
with h5py.File(simulation_name, "r") as simulation_results:
    fig, plot_func = plot_all(simulation_results)
    plot_func(-1)
    fig.suptitle(f"{d=} {chi=} {chi_PC=}")
    fig
#simulation_results.close()
#%%
#fig.savefig("result2.svg")
#%%
simulation_results = h5py.File(simulation_name, "r")
#%%
import matplotlib.animation as animation
fig, plot_func = plot_all(simulation_results)
ani = animation.FuncAnimation(fig, plot_func, frames=len(simulation_results["timestep"]))
FFwriter = animation.FFMpegWriter(fps = 25)
#%%
mp4_name = simulation_name.replace("simulation_data/", "media/").replace(".h5",".mp4")
ani.save(mp4_name, writer=FFwriter)
# %%
nsteps = len(simulation_results["timestep"])
J_tot_in = [simulation_results["J_z_tot"][i][1] for i in range(nsteps)]
J_tot_out = [simulation_results["J_z_tot"][i][-1] for i in range(nsteps)]


fig, ax = plt.subplots()
ax.plot(simulation_results["timestep"][1:], J_tot_in[1:], label = "$J_{in}$")
ax.plot(simulation_results["timestep"][1:], J_tot_out[1:], label = "$J_{out}$")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("$\int^{r_{pore}} j(r,z) dr$")
#%%
fig, ax = plt.subplots()
ax.plot(simulation_results["timestep"][10:], np.array(J_tot_in[10:])/np.array(J_tot_out[10:]), label = r"$\frac{J_{in}}{J_{out}}$")
ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("$\int^{r_{pore}} j(r,z) dr$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid()
# %%
c_total = [float(np.sum(simulation_results["c_z_average"][i])) for i in range(nsteps)]
fig, ax = plt.subplots()
ax.plot(simulation_results["timestep"][:], c_total)
#ax.set_xscale("log")
#ax.set_yscale("log")
ax.set_xlabel("t")
ax.set_ylabel("$c_{total}$")
ax.grid()
#simulation_results.close()
# %%
def reverse_gradient(gradient):
    # Initialize the original array with zeros
    shape = np.shape(gradient[0])
    original_array = np.zeros(shape)
    
    # Integrate the gradient values to reconstruct the original array
    for axis in [0]:
        original_array += np.cumsum(gradient[axis], axis=axis)

    return original_array

def calculate_divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])

def calculate_curl(vector_field):
    # Ensure that the input vector_field has the shape (2, m, n) where m and n are the dimensions of the grid.
    if vector_field.shape[0] != 2:
        raise ValueError("Input vector_field must have shape (2, m, n).")

    # Calculate the partial derivatives
    dF2_dx = np.gradient(vector_field[1], axis=1)
    dF1_dy = np.gradient(vector_field[0], axis=0)

    # Calculate the curl
    curl = dF2_dx - dF1_dy

    return curl
#%%
#plot_heatmap_and_profiles(div_j)
# %%
