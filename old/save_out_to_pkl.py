#%%
import pickle
import sfbox_utils
from typing import Union
import pathlib
import uuid
#%%

#%%
def save_output_as_pickle(filename : Union[str, pathlib.Path],  pickle_name=None, dir=None, **kwargs):
    filename = pathlib.Path(filename)
    if pickle_name is None:
        pickle_name = filename.with_suffix(".pkl").name
    else:
        pickle_name = pathlib.Path(pickle_name).name
    if dir is None:
        dir = filename.parent
    data = sfbox_utils.read_output.parse_file(filename, **kwargs)
    print(dir/pickle_name)
    with open(dir/pickle_name, "wb") as f:
        pickle.dump(data, f)

# %%
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
path = pathlib.Path().parent
dir = path/"data"/"raw"/"particle_insertion"
for file in (path/"temp2"/"split").glob("*.out"):
    #pkl_name = str(uuid.uuid4()) + ".pkl"
    save_output_as_pickle(file, dir = dir, read_fields = read_fields, read_fields_regex=read_fields_regex)
    break
#%%
