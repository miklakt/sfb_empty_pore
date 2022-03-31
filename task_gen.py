#%%
import sfbox_utils
import numpy as np
from tenacity import retry
#%%
template =  sfbox_utils.read_input.parse_file("template.txt")

def set_system_geometry(r, s, h, l1, l2):
    xlayers = r+h
    ylayers = l1+s+l2

    x1 = r+1
    x2 = r+h
    y1 = l1+1
    y2 = l1+s
    frozen_range = f"{x1},{y1};{x2},{y2}"

    d = {
    'lat:2G:n_layers_x':xlayers,
    'lat:2G:n_layers_y':ylayers,
    'mon:W:frozen_range': frozen_range,
    }

    return d

def set_polymer_brush(N, theta, r, s, l1):
    x1 = r+1
    y1 = l1+1
    y2 = l1+s
    compositions = [f"(P{i})1(P){N-1}" for i in range(s)]
    pinned_ranges = [f"{x1-1},{y};{x1-1},{y}" for y in range(y1,y2+1)]

    d ={}
    d.update(
        {f'mon:P{i}:freedom': 'pinned' for i in range(s)}
    )
    d.update(
        {f'mol:pol{i}:freedom': 'restricted' for i in range(s)}
    )
    d.update(
        {f'mol:pol{i}:theta': theta for i in range(s)}
    )

    d.update(
        {f'mon:P{i}:pinned_range': pinned_range for i, pinned_range in enumerate(pinned_ranges)}
    )
    d.update(
        {f'mol:pol{i}:composition': composition for i, composition in enumerate(compositions) }
    )
    return d

def set_chi(chi_PW, chi_PS, s):

    d={f'mon:P{i}:chi - W': chi_PW for i in range(s)}

    d.update(
        {f'mon:P{i}:chi - S': chi_PS for i in range(s)}
    )

    d.update(
        {
        f'mon:P:chi - W': chi_PW,
        f'mon:P:chi - S': chi_PS,
        }
    )

    return d

def args_to_name(args, ignore_keys = []):
    return "_".join([str(k) + "=" + str(v) for k, v in args.items() if k not in ignore_keys])

def generate_task_empty_pore(r, s, h, l1, l2, chi_PW, chi_PS, N, theta, **kwargs):
    d ={}

    d = set_system_geometry(r,s,h,l1,l2)

    d.update(set_polymer_brush(N, theta, r, s, l1))

    d.update(set_chi(chi_PW, chi_PS, s))

    return d
# %%

args = dict(
    r = 52,
    s = 52,
    h = 40,
    l1 = 120,
    l2 = 120,
    chi_PS = 0.1,
    chi_PW = 0,
    N=300,
    sigma = 0.02
)
args["theta"] =  2*np.pi*args["r"]*args["N"]*args["sigma"]
task = generate_task_empty_pore(**args)

template[0].update(task)

fname = args_to_name(args, ignore_keys=["theta"])+".in"
#%%
sfbox_utils.write_input.write_input_file(fname, template)
# %%
sfbox_utils.sfbox_calls(parallel_execution='subprocess')
# %%
