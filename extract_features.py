#%%
from typing import Dict, List
import numpy as np
import sfbox_utils

def particle_geometry(calculation):
    x0, y0, x1, y1 = list(map(int,calculation['mon:C:frozen_range'].replace(";", ",").split(',')))
    h = y1-y0+1
    c = (y1+y0-1)/2
    w = 2*x1
    return {"pc" : int(c), "ph" : int(h), "pw" : int(w)}

def pore_geometry(calculation):
    xlayers =  calculation["lat:2G:n_layers_x"]
    ylayers =  calculation["lat:2G:n_layers_y"]

    frozen_range = list(map(int,calculation['mon:W:frozen_range'].replace(";", ",").split(',')))

    r = frozen_range[0]-1
    l1 = frozen_range[1] -1
    s = frozen_range[3] - frozen_range[1]+1
    h = xlayers - r
    l2 = ylayers - l1 - s

    N = calculation['mol:pol0:chainlength']
    theta = calculation['mol:pol0:theta']*s
    sigma = theta/(2*np.pi*r*N)/s

    data = {
        "l1" : int(l1),
        "l2" : int(l2),
        "s" : int(s),
        "r" : int(r),
        "h" : int(h),
        "xlayers" : int(xlayers),
        "ylayers" : int(ylayers),
        "N" : int(N),
        "theta" : float(theta),
        "sigma" : float(sigma)
    }
    return data

def polymer_density(calculation, xlayers, ylayers, s, **_):
    phi_P = calculation["mon:P:phi:profile"].reshape((xlayers,ylayers))
    phi_0 = np.sum([calculation[f"mon:P{i}:phi:profile"].reshape((xlayers,ylayers)) for i in range(s)], axis =0)
    phi = phi_0+phi_P
    return {"phi" : phi, "grafting_phi" : phi_0}

def strands_density(calculation, xlayers, ylayers, s, **_):
    data = {}
    data.update({f"pol{i}_phi" : calculation[f"mol:pol{i}:phi:profile"].reshape((xlayers,ylayers)) for i in range(s)})
    return data

def polymer_potential(calculation, xlayers, ylayers, s, **_):
    data = {"P_G":calculation["mon:P:G:profile"].reshape((xlayers,ylayers))}
    data.update({"S_G":calculation["mon:S:G:profile"].reshape((xlayers,ylayers))})
    data.update({f"P{i}_G" : calculation[f"mon:P{i}:G:profile"].reshape((xlayers,ylayers)) for i in range(s)})
    return data

def chi_params(calculation):
    if 'chi list:P:chi - C' in calculation.keys():
        d = {   "chi_PC" : calculation['chi list:P:chi - C']    }
    else:
        d = {}

    d.update({  "chi_PW" : calculation['chi list:P:chi - W'],
                "chi_PS" : calculation['chi list:P:chi - S']    })
    return d

def free_energy(calculation):
    return {"free_energy" : calculation['sys:noname:free energy']}

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

    if save_to:
        sfbox_utils.write_initial_guess("initial_guess.ig", initial_guess_dict)

    return initial_guess_dict
# %%
