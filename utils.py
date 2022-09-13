#%%
from typing import Dict, List
import pandas as pd
import numpy as np
import pandas as pd
import sfbox_utils
import itertools

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def check_if_exist(dataframe, **kwargs):
    return all([any(dataframe[k] == v) for k,v in kwargs.items()])

def get_by_kwargs(dataframe, **kwargs):
    return dataframe.loc[(dataframe[list(kwargs)] == pd.Series(kwargs)).all(axis=1)]

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

def load_datasets(df : pd.DataFrame, keys : List):
    for key in keys:
        df[key] = np.nan
        df[key] = df[key].astype(object)
        df[key] = df.apply(lambda _: sfbox_utils.store.load_dataset(_.h5file, f"/{key}"), axis=1)