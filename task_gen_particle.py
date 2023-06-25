#%%
import pathlib
import sfbox_utils
import numpy as np
import pandas as pd
import itertools
import utils
#%%
ref_tbl = pd.read_pickle("reference_table.pkl")

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

def set_particle_position_absolute(pc : int, ph : int, pw : int, **_):
    frozen_range = str_frozen_range(pc, ph, pw)
    d = {
    'mon:C:frozen_range': frozen_range,
    }
    return d

def set_particle_position(pc : int, ph : int, pw : int, l1: int, s:int, **_):
    pc_absolute = int(pc + l1 + s/2)
    return set_particle_position_absolute(pc_absolute, ph, pw)

def translation_chain(builder_funcs, **kwargs):
    d ={}
    for func in builder_funcs:
        d.update(func(**kwargs))
    return d

def translate_to_sfbox(data):
    init_keys = data[0]
    
    init_statements_builder = [
        set_system_geometry,
        set_polymer_brush,
        set_chi,
        set_particle_position
    ]
    
    followed_statements_builder = [
        set_particle_position
    ]

    init_statement =  sfbox_utils.read_input.parse_file("sfb_templates/template_particle.txt")[0]
    
    init_statement.update(
        translation_chain(init_statements_builder, **init_keys))

    followed_statements = []

    for followed_keys in data[1:]:
        keys = dict(init_keys)
        keys.update(followed_keys)
        followed_statement = translation_chain(
            followed_statements_builder, **keys
            )
        followed_statements.append(followed_statement)

    return [init_statement] + followed_statements

def sort_keys(list_of_dicts):
    for i, d in enumerate(list_of_dicts):
        list_of_dicts[i] = dict(sorted(d.items()))
    return list_of_dicts

def args_to_name(args, ignore_keys = []):
    return "_".join([str(k) + "=" + str(v) for k, v in args.items() if k not in ignore_keys])

def move_along_z(
        init_args, 
        pc_ground_energy, 
        pc_start, 
        pc_end, 
        pc_step=1, 
        skip_found = True
        ):
    pc_list = [pc_ground_energy] + list(range(pc_start, pc_end, pc_step))
    if 0 not in pc_list: pc_list.append(0)
    pc_list.sort()

    ifile_data = []

    skipped = 0
    for pc in pc_list:
        record_in_reference = utils.get_by_kwargs(ref_tbl, pc = pc, **init_args)
        
        if len(record_in_reference) == 0 or not(skip_found):
            ifile_data.append(dict(pc = pc))
        
        else:
            #print(f"The calculation is skipped {pc=}")
            skipped = skipped+1

    if ifile_data:
        ifile_data[0].update(init_args)

    print(f"Calculation skipped: {skipped}")

    return ifile_data

def grow_particle(init_args, ph_start, ph_end, skip_found = False):
    ifiles_data = []
    fnames = []
    ifile_data = []
    init = True
    ph_list = list(range(ph_start, ph_end+2, 2))
    for ph in ph_list:
        if len(
            utils.get_by_kwargs(
                ref_tbl, ph = ph, pw=ph, **init_args
                    )
                ) == 0 or not(skip_found):
            if init:
                ifile_data.append(
                    dict(**init_args, ph = ph, pw = ph)
                )
                init = False
            else:
                ifile_data.append(dict(ph = ph, pw = ph))
    
    if ifile_data:
        ifiles_data.append(ifile_data)
        fname = args_to_name(ifile_data[0], ignore_keys=["ph"])+".in"
        fnames.append(fname)
    ifiles = [
        create_input_file_data(i) for i in ifiles_data
    ]
    return ifiles, fnames
            
#%%
init_args = dict(
        r = 26,
        s = 52,
        h = 40,
        l1 = 120,
        l2 = 120,
        chi_PW = 0,
        chi_PC = 0.0,
        N=300,
        sigma = 0.02,
        ph = 4,
        pw = 4,
        chi_PS = 0.4
)
working_dir = path = pathlib.Path("temp8")
continue_unfinished = True
try:
    working_dir.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print("Folder is already there")
#%%
ifile_data = move_along_z(init_args, -126, -80, 80)
translated = translate_to_sfbox(ifile_data)
translated = sort_keys(translated)

if continue_unfinished:
    ig_data = utils.find_closest_in_reference(
        ref_tbl,
        init_args | {"pc" : ("close", ifile_data[0]["pc"])},
        return_only_one = True
        )
    if (ig_data is not None) and (not ig_data.empty):
        #print("Initial guess is found")
        pass
    else:
        ig_data = False

input_file = working_dir/(args_to_name(ifile_data[0], ignore_keys=["pc", "initial_guess"]) + ".in")
initial_guess_file = input_file.with_suffix(".ig")
if ig_data is not False: 
    utils.calculation_result_to_initial_guess_file(ig_data, initial_guess_file)

if initial_guess_file.exists():
    print("Initial guess file is found")
    translated[0].update({"newton:isaac:initial_guess":"file"})
    translated[0].update({"newton:isaac:initial_guess_input_file":"filename.ig"})
    try:
        translated[1].update({"newton:isaac:initial_guess":"previous_result"})
    except IndexError:
        pass

sfbox_utils.write_input.write_input_file(input_file, translated)

print(f"Number of calculations: {len(translated)}")
# %%
