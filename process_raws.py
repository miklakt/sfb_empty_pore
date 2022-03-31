#%%
import pandas as pd
import pathlib
import pickle
#%%
data = []
for file in pathlib.Path("data/raw").glob("*.pkl"):
    r = pickle.load(open(file, "rb"))
    r["filename"] = file.name
    data.append(r)
#%%
df = pd.DataFrame(data)
rename_columns = {
    'chi list:P:chi - W' : 'chi_PW',
    'chi list:P:chi - S' : 'chi_PS',
    'mol:pol0:chainlength' : 'N',
    'mol:pol0:theta' : 'theta_per_unit_length',
    'mon:P:phi:profile' : 'phi',
    "lat:2G:n_layers_x" : 'xlayers',
    "lat:2G:n_layers_y" : 'ylayers',
    "filename" : 'filename'
}
df=df[rename_columns.keys()]
df.rename(columns = rename_columns, inplace = True)
#%%
df.to_pickle("data/empty_brush.pkl")
# %%
