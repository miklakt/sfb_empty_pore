#%%
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
    selected = dataframe

    for key, value in kwargs.items():
        #print (key)
        if selected.empty: return selected
        
        if isinstance(value, list):
            selected = selected.loc[selected[key].isin(value)]
            continue
    
        if isinstance(value, tuple):
            op = value[0]
            arg = value[1]
            if op in ["!=", ">=", ">", "<=", "<"]:
                eval_str = fr"selected.loc[selected['{key}']{op}{arg}]"
                selected = eval(eval_str)
                continue
            
            elif op == "close":
                value = (selected[key] - arg).abs().idxmin()
                selected = selected.loc[value]
                continue
            else:
                raise ArithmeticError("No operation defined")
            
        if value == "smallest":
            value = selected[key].min()
        if value == "largest":
            value = selected[key].max()
        
        selected = selected.loc[selected[key]==value]

    return selected


def calculation_result_to_initial_guess_file(calc_result, save_to = None):
    xlayers = calc_result.xlayers
    ylayers = calc_result.ylayers
    gradients = (2, xlayers+2, ylayers+2)
    s = calc_result.s
    phibulk_solvent = 1
    molecules = {"P" : calc_result.dataset["P_G"], "S" : calc_result.dataset["S_G"]}
    molecules.update({f"P{i}" : calc_result.dataset[f"P{i}_G"] for i in range(s)})
    for k, v in molecules.items():
        molecules[k] = np.pad(v.squeeze(), ((1,1),(1,1)), "edge")
    alphabulks = {k : 1.0 for k in molecules.keys()}

    initial_guess_dict = {"state" : molecules, "phibulk solvent" : phibulk_solvent, "alphabulk" : alphabulks, "gradients" : gradients}
    if save_to is None:
        sfbox_utils.write_initial_guess("initial_guess.ig", initial_guess_dict)
    else:
        sfbox_utils.write_initial_guess(save_to, initial_guess_dict)
    return initial_guess_dict
# %%

def find_closest_in_reference(reference_tbl, requires_dict, optional_dict = {}, return_only_one = False):
    df = get_by_kwargs(reference_tbl, **requires_dict)
    if df.empty:
        return None
    if len(df)==1:
        return df

    optional_df = df
    for key in optional_dict.keys():
        if len(optional_df)==1:
            print(f"last optional key used: {key}")
            break
        new_optional_df = get_by_kwargs(optional_df, **{key : optional_dict[key]})
        if len(new_optional_df) == 0:
            break
        optional_df = new_optional_df

    if return_only_one:
        if not isinstance(optional_df, pd.Series):
            raise KeyError("Too many records, try too concretize search arguments")
    
    return optional_df
# %%
