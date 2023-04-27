#%%
import sfbox_utils
import numpy as np
import matplotlib.pyplot as plt
import extract_features
import pandas as pd
# %%
fname = "r=26_s=52_h=40_l1=120_l2=120_chi_PS=1.1_chi_PW=0_N=300_sigma=0.02"
data = next(sfbox_utils.read_output.parse_file(
    f"temp/{fname}.out",
    #read_fields=["mon : P : phi : profile",
    #             "lat : 2G : n_layers_x",
    #            "lat : 2G : n_layers_y",],
    ))
#%%
import pickle
with open(f'empty_pore_pickles/{fname}.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
features = extract_features.pore_geometry(data)
features.update(extract_features.polymer_density(data, **features))
features.update({
        "chi_PW" : data['chi list:P:chi - W'],
        "chi_PS" : data['chi list:P:chi - S'],
        "free_energy" : data['sys:noname:free energy'],
        })
features['theta_per_unit_length'] = features['theta']/features['s']
features["filename"] = f"{fname}.out"

#%%
dataset = pd.read_pickle("empty_brush.pkl")
# %%
to_append = pd.DataFrame([{k:features[k] for k in dataset.columns}])
# %%
dataset = pd.concat([dataset, to_append], ignore_index=True)
# %%
pd.to_pickle(dataset, "empty_brush.pkl")
# %%
