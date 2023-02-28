#%%
from typing import Dict, List
import numpy as np

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
    return {"phi" : phi, "phi_0" : phi_0}
