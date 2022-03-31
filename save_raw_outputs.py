#%%
import pickle
import sfbox_utils
from typing import Union
import pathlib
import uuid
#%%
def save_output_as_pickle(filename : Union[str, pathlib.Path],  pickle_name=None, dir=None):
    filename = pathlib.Path(filename)
    if pickle_name is None:
        pickle_name = filename.with_suffix("pkl")
    else:
        pickle_name = pathlib.Path(pickle_name)
    if dir is None:
        dir = filename.parent
    data = sfbox_utils.read_output.parse_file(filename)
    with open(dir/pickle_name, "wb") as f:
        pickle.dump(data, f)

# %%
for file in pathlib.Path().glob("*.out"):
    pkl_name = str(uuid.uuid4()) + ".pkl"
    pkl_name = file.with_suffix(".pkl")
    dir = pathlib.Path("data/raw")
    print(pkl_name)
    save_output_as_pickle(file, pkl_name, dir)
    print(f"{file} saved as {pkl_name}")
# %%
