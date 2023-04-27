#%%
import pathlib
import sfbox_utils
from extract_features import polymer_density, pore_geometry, particle_geometry

dir  =  "1405"
read_fields = [
    "lat : 2G : n_layers_x",
    "lat : 2G : n_layers_y",
    "mon : W : frozen_range",
    "mon : C : frozen_range",
    "mon : P : phi : profile",
    'chi list : P : chi - W',
    'chi list : P : chi - S',
    'chi list : P : chi - C',
    'mol : pol0 : chainlength',
    'mol : pol0 : theta',
    'sys : noname : free energy'
]
read_fields_regex = [
    "mon : P([0-9]|[0-9][0-9]) : phi : profile"
]

def process_data(raw_data):
    data = pore_geometry(raw_data)
    data.update(polymer_density(raw_data, **data))
    data.update(particle_geometry(raw_data))
    data.update({
        "chi_PC" : raw_data['chi list:P:chi - C'],
        "chi_PW" : raw_data['chi list:P:chi - W'],
        "chi_PS" : raw_data['chi list:P:chi - S'],
        "free_energy" : raw_data['sys:noname:free energy'],
        })
    data["sigma"] = round(data["sigma"], 4)
    return data

def name_file(data):
    keys = ["chi_PS", "chi_PC", "pc", "ph", "pw", "r", "xlayers", "ylayers", "N", "sigma"]
    name = "_".join([f"{k}_{data[k]}" for k in keys])
    return name

#%%
for file in pathlib.Path(dir).glob("*.out"):
    print(file.name)
    sfbox_utils.store.store_file_parallel(
        file,
        dir = "h5",
        process_routine=process_data,
        naming_routine=name_file,
        n_jobs=12,
        reader_kwargs=dict(
            read_fields = read_fields,
            read_fields_regex = read_fields_regex
        )
    )

# %%
