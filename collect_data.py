#%%
import numpy as np
import pathlib
import sfbox_utils
from extract_features import polymer_density, pore_geometry
from extract_features import particle_geometry, polymer_potential, strands_density
from extract_features import chi_params, free_energy
from extract_features import ground_energy_correction

read_fields = [
    "lat : 2G : n_layers_x",
    "lat : 2G : n_layers_y",
    "mon : W : frozen_range",
    "mon : C : frozen_range",
    "mon : P : phi : profile",
    "mon : P : G : profile",
    "mon : S : G : profile",
    'chi list : P : chi - W',
    'chi list : P : chi - S',
    'chi list : P : chi - C',
    'mol : pol0 : chainlength',
    'mol : pol0 : theta',
    'sys : noname : free energy'
]
read_fields_regex = [
    "mon : P([0-9]|[0-9][0-9]|[0-9][0-9][0-9]) : phi : profile",
    "mon : P([0-9]|[0-9][0-9]|[0-9][0-9][0-9]) : G : profile",
    "mol : pol([0-9]|[0-9][0-9]|[0-9][0-9][0-9]) : phi : profile"
]


def process_data(raw_data):
    data = pore_geometry(raw_data)
    data.update(polymer_density(raw_data, **data))
    data.update(polymer_potential(raw_data, **data))
    data.update(strands_density(raw_data, **data))
    data.update(particle_geometry(raw_data))
    data.update(chi_params(raw_data))
    data.update(free_energy(raw_data))
    data["sigma"] = round(data["sigma"], 4)
    data["comment"] = "moved_along_z"
    return data

def name_file(data, timestamp = True):
    keys = ["chi_PS", "chi_PC", "pc", "ph", "pw", "r", "xlayers", "ylayers", "N", "sigma"]
    name = "_".join([f"{k}_{data[k]}" for k in keys])
    if data.get("comment"):
        name = "_".join([data['comment'], name])
    #if timestamp:
    #    import time
    #    timestamp = time.strftime("%Y%m%d-%H%M%S")
    #    name = f"{name}_{timestamp}"
    return name

#%%
dir_ = "temp230624"
for file in pathlib.Path(dir_).glob("*.out"):
    print(file)
    if file:
        sfbox_utils.store.store_file_parallel(
            file,
            dir = "h5",
            process_routine=process_data,
            naming_routine=name_file,
            n_jobs=12,
            reader_kwargs=dict(
                read_fields = read_fields,
                read_fields_regex = read_fields_regex
            ),
            on_file_exist="keep",
            on_process_error = "raise",
        )
# %%
def ground_energy_correction(df, group_columns):
    df.sort_values(by = group_columns, inplace = True)
    df.reset_index(drop = True, inplace=True)
    for idx, group in df.groupby(by =group_columns):
        y_max=group['pc'].min()
        energy_0 = group.loc[group['pc'] == y_max, 'free_energy'].squeeze()
        if not(isinstance(energy_0, float)):
            raise ValueError("Ground energy cannot be 'float'\n" \
                    + f"idx: {idx}, ymax: {y_max}, value: {energy_0}, type: {type(energy_0)}")
        df.loc[group.index, 'free_energy'] = df.loc[group.index, 'free_energy'] - energy_0

master = sfbox_utils.store.create_reference_table(storage_dir="h5")
group_by = [
    'N', 'chi_PC', 'chi_PS', 'chi_PW',
    'h', 'l1', 'l2', 'ph',
    'pw','r', 's', 'sigma'
    ]
ground_energy_correction(master, group_by)
master.pc = master.pc - (master.l1+master.s/2)
master.pc = master.pc.astype(int)
master.to_pickle("reference_table.pkl")
# %%