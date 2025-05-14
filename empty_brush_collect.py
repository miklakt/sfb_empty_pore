#%%
import sfbox_utils
import numpy as np
import matplotlib.pyplot as plt
import extract_features
import pandas as pd
import pathlib
#%%
dir_ = pathlib.Path("temp0303")
#%%
def process_data(raw_data):
    import extract_features
    data = extract_features.pore_geometry(raw_data)
    data.update(extract_features.polymer_density(raw_data, **data))
    data.update(extract_features.polymer_potential(raw_data, **data))
    data.update(extract_features.strands_density(raw_data, **data))
    data.update(extract_features.chi_params(raw_data))
    data.update(extract_features.free_energy(raw_data))
    data["sigma"] = round(data["sigma"], 4)
    data["comment"] = "independent_run"
    return data

def name_file(data, timestamp = True):
    keys = ["chi_PS", "r", "xlayers", "ylayers", "N", "sigma"]
    name = "_".join([f"{k}_{data[k]}" for k in keys])
    if data.get("comment"):
        name = "_".join([data['comment'], name])
    return name
# %%
for filename in dir_.glob("*.out"):
    print(filename)
    sfbox_utils.store.store_file_sequential(
        file=filename,
        process_routine=process_data,
        naming_routine=name_file,
        dir = "h5_empty_pore",
        on_file_exist="rename",
    )
# %%
master = sfbox_utils.store.create_reference_table(storage_dir="h5_empty_pore")
# %%
master.to_pickle("pkl/reference_table_empty_brush.pkl")
# %%
