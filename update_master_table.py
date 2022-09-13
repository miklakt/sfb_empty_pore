import numpy as np
import sfbox_utils
import h5py
import pandas
import utils

master = sfbox_utils.store.create_master_table(dir = "h5")
group_by = [
    'N', 'chi_PC', 'chi_PS', 'chi_PW', 
    'h', 'l1', 'l2', 'ph', 
    'pw','r', 's', 'sigma'
    ]
utils.ground_energy_correction(master, group_by)
master.to_pickle("reference_table.pkl")