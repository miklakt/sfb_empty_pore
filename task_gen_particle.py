#%%
import pathlib
import sfbox_utils
import numpy as np
import pandas as pd
import itertools
import utils
#%%
ref_tbl = pd.read_pickle("reference_table.pkl")
working_dir = path = pathlib.Path("1405")


def set_system_geometry(r, s, h, l1, l2, **_):
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

def set_polymer_brush(N, sigma, r, s, l1, **_):
    theta_per_ring =  2*np.pi*r*N*sigma
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
        {f'mol:pol{i}:theta': theta_per_ring for i in range(s)}
    )

    d.update(
        {f'mon:P{i}:pinned_range': pinned_range for i, pinned_range in enumerate(pinned_ranges)}
    )
    d.update(
        {f'mol:pol{i}:composition': composition for i, composition in enumerate(compositions) }
    )
    return d

def set_chi(chi_PW, chi_PS, chi_PC, s, **_):

    d={f'mon:P{i}:chi - W': chi_PW for i in range(s)}

    d.update(
        {f'mon:P{i}:chi - S': chi_PS for i in range(s)}
    )

    d.update(
        {f'mon:P{i}:chi - C': chi_PC for i in range(s)}
    )

    d.update(
        {
        f'mon:P:chi - W': chi_PW,
        f'mon:P:chi - S': chi_PS,
        f'mon:P:chi - C': chi_PC,
        }
    )

    return d

def str_frozen_range(pc : int, ph : int, pw : int, **_) -> str:
    """The particle size and position in sfbox are described with a frozen_range (see sfbox manual)

    Args:
        y0 (int): Position of particle center
        h (int): Particle's height
        w (int): Particle's width

    Returns:
        (str): String to use in sfbox input file
    """
    if (pw % 2 != 0) or (ph % 2 != 0):
        print('Odd values are not implemented yet')
        raise ValueError('Odd values are not implemented yet')
    _str=f'1,{int(pc-ph/2+1)};{int(pw/2)},{int(pc+ph/2)}'
    return _str

def set_particle(pc : int, ph : int, pw : int, **_):
    frozen_range = str_frozen_range(pc, ph, pw)
    d = {
    'mon:C:frozen_range': frozen_range,
    }
    return d

def user_keys_to_sfb_keywords(builder_funcs, **kwargs):
    d ={}
    for func in builder_funcs:
        d.update(func(**kwargs))
    return d


def create_input_file_data(data):
    init_keys = data[0]
    init_statements_builder = [
        set_system_geometry, 
        set_polymer_brush, 
        set_chi, 
        set_particle
    ]
    followed_statements_builder = [
        set_particle
    ]

    init_statement =  sfbox_utils.read_input.parse_file("sfb_templates/template_particle.txt")[0]
    init_statement.update(user_keys_to_sfb_keywords(init_statements_builder, **init_keys))

    #followed_statements_keys = ""
    followed_statements = []
    
    for followed_keys in data[1:]:
        keys = dict(init_keys)
        keys.update(followed_keys)
        followed_statement = user_keys_to_sfb_keywords(
            followed_statements_builder, **keys
            )
        followed_statements.append(followed_statement)

    return [init_statement] + followed_statements

def args_to_name(args, ignore_keys = []):
    return "_".join([str(k) + "=" + str(v) for k, v in args.items() if k not in ignore_keys])

#%%
init_args = dict(
        r = 26,
        s = 52,
        h = 40,
        l1 = 120,
        l2 = 120,
        chi_PW = 0,
        chi_PC = -1.5,
        N=300,
        sigma = 0.02,
        ph = 4, 
        pw = 4,
        #chi_PS = ...
        #pc = ...
)
step = 3
ylayers = init_args["l1"]+init_args["s"]+init_args["l2"]
pc_ground_energy = 20
pc_list = [pc_ground_energy] + list(range(48+step, int(ylayers/2)+1, step)) + [int(ylayers/2)]
chi_ps_list = [0.3, 0.4, 0.5, 0.6, 0.7]

ifiles_data = []
fnames = []
for chi_ps in chi_ps_list:
    ifile_data = []
    init = True
    for pc in pc_list:
        if len(
            utils.get_by_kwargs(
                ref_tbl, chi_PS = chi_ps, pc = pc, **init_args
                    )
                ) == 0:
            print(chi_ps, pc)
            if init:
                ifile_data.append(
                    dict(**init_args, chi_PS = chi_ps, pc = pc)
                )
                init = False
            else:
                ifile_data.append(dict(pc = pc))
    if ifile_data:
        ifiles_data.append(ifile_data)
        fnames.append(args_to_name(ifile_data[0])+".in")

ifiles = [
    create_input_file_data(i) for i in ifiles_data
]
# %%
for file_lines, fname in zip(ifiles, fnames):
    try:
        working_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")
    sfbox_utils.write_input.write_input_file(working_dir/fname, file_lines)
# %%
sfbox_utils.sfbox_calls(parallel_execution='subprocess', dir = working_dir)
# %%
