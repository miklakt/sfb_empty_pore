#%%
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

#import calculate_fields_in_pore

def pad_fields(fields, pad_sides, pad_top):
    #pad fields to the sides and to the outer radius
    fields["xlayers"]=fields["xlayers"]+pad_top
    fields["ylayers"]=fields["ylayers"]+pad_sides*2

    fields["h"]=fields["h"]+pad_top
    fields["l1"]=fields["l1"]+pad_sides
    fields["l2"]=fields["l2"]+pad_sides
    padding = ((pad_sides, pad_sides),(0, pad_top))

    for k in fields.keys():
        if k in ["walls", "mobility", "conductivity"]: continue
        try:
            fields[k] = np.pad(
                fields[k],
                padding, 
                "constant", constant_values=(0.0, 0.0)
                )
            #print(k, "padded")
        except ValueError:
            pass
        
    fields["walls"]=np.pad(
        fields["walls"],
        padding,
        "edge",
        )
    #print("walls", "padded")
    
    fields["mobility"]=np.pad(
        fields["mobility"],
        padding, 
        "constant", constant_values=(1.0, 1.0)
        )
    fields["mobility"][fields["walls"]==True]=0.0
    #print("mobility", "padded")

    bulk = fields["conductivity"][1,1]
    fields["conductivity"]=np.pad(
        fields["conductivity"],
        padding, 
        "constant", constant_values=(bulk, bulk)
    )
    fields["conductivity"][fields["walls"]==True]=0.0
    #print("conductivity", "padded")

def is_inside_ellipse(x, y, a, b, x0=0.0, y0=0.0, side='full'):
    """
    Check whether (x, y) is inside a full or half ellipse centered at (x0, y0),
    with focal points aligned along the y-axis and separated by distance a,
    and semi-minor axis b.

    Parameters:
        x, y : np.ndarray or float — coordinates to evaluate
        a : float — distance between focal points (along y-axis)
        b : float — semi-minor axis
        x0, y0 : float — center of the ellipse
        side : str — 'full', 'left', or 'right' half

    Returns:
        Boolean array or value: True if inside specified portion of the ellipse
    """
    f1 = np.array([x0, y0 - a / 2])
    f2 = np.array([x0, y0 + a / 2])
    p = np.stack([x, y], axis=-1)

    d1 = np.linalg.norm(p - f1, axis=-1)
    d2 = np.linalg.norm(p - f2, axis=-1)

    c = np.sqrt(b**2 + (a / 2)**2)
    inside = (d1 + d2) <= 2 * c

    if side == 'left':
        return inside & (x <= x0)
    elif side == 'right':
        return inside & (x >= x0)
    else:  # 'full'
        return inside
#%%
def R_steady_state(fields):
    l1 = fields["l1"]
    pore_radius = fields["r"]
    pad_sides = 100
    pad_top = np.sqrt((l1)**2 + (pore_radius / 2)**2)
    #pad_top = np.sqrt(4*(l1+pad_sides)**2 + pore_radius**2)
    #pad_fields(fields, pad_sides, pad_top)

    Nr, Nz = fields["xlayers"], fields["ylayers"]
    dr, dz = 1,1
    # Grid
    r = np.arange(0, Nr)
    z = np.arange(0, Nz)
    R_grid, Z_grid = np.meshgrid(r, z, indexing='ij')

    sigma = fields["conductivity"][Z_grid, R_grid]
    R_local =sigma**-1

    l1 = fields["l1"]
    mask_z = z<l1
    mask = mask_z[:, np.newaxis]
    bc_source = ~is_inside_ellipse(
        Z_grid, 
        R_grid, 
        a = fields["r"], 
        b=l1+1, 
        x0=l1+1,
        side = "left"
        ).T*mask
    bc_source[fields["walls"]==True] = False

    
    bc_sink = np.zeros((Nr, Nz), dtype=bool)
    bc_sink = bc_source[::-1]

    # Dirichlet boundary conditions
    psi_source = 1.0
    psi_sink = 0.0

    # Helper to map 2D (i,j) to 1D index
    def idx(i, j):
        return i * Nz + j

    # Build sparse matrix A for the operator
    data, rows, cols = [], [], []
    b = np.zeros(Nr * Nz)

    for i in range(Nr):
        for j in range(Nz):
            k = idx(i, j)
            r_i = r[i]

            if bc_source[j, i]:
                rows.append(k)
                cols.append(k)
                data.append(1.0)
                b[k] = psi_source
                continue
            elif bc_sink[j, i]:
                rows.append(k)
                cols.append(k)
                data.append(1.0)
                b[k] = psi_sink
                continue
            elif not np.isfinite(R_local[i, j]):
                rows.append(k)
                cols.append(k)
                data.append(1.0)
            elif i == 0:
                if np.isfinite(R_local[i+1, j]):
                    kp = idx(i + 1, j)
                    rows.extend([k, k])
                    cols.extend([k, kp])
                    data.extend([-1 / dr, 1 / dr])
                else:
                    rows.append(k)
                    cols.append(k)
                    data.append(1.0)
            elif i == Nr - 1:
                if np.isfinite(R_local[i-1, j]):
                    km = idx(i - 1, j)
                    rows.extend([k, k])
                    cols.extend([k, km])
                    data.extend([-1 / dr, 1 / dr])
                else:
                    rows.append(k)
                    cols.append(k)
                    data.append(1.0)
            else:
                neighbors_valid = {
                    'rp': np.isfinite(R_local[i + 1, j]),
                    'rm': np.isfinite(R_local[i - 1, j]),
                    'zp': np.isfinite(R_local[i, j + 1]),
                    'zm': np.isfinite(R_local[i, j - 1]),
                }

                coeff_center = 0
                stencil = []

                if neighbors_valid['rp']:
                    sr_p = (sigma[i + 1, j] + sigma[i, j]) / 2
                    stencil.append((idx(i + 1, j), sr_p / dr**2 + sr_p / (2 * dr * r_i)))
                    coeff_center -= sr_p / dr**2 + sr_p / (2 * dr * r_i)

                if neighbors_valid['rm']:
                    sr_m = (sigma[i - 1, j] + sigma[i, j]) / 2
                    stencil.append((idx(i - 1, j), sr_m / dr**2 - sr_m / (2 * dr * r_i)))
                    coeff_center -= sr_m / dr**2 - sr_m / (2 * dr * r_i)

                if neighbors_valid['zp']:
                    sz_p = (sigma[i, j + 1] + sigma[i, j]) / 2
                    stencil.append((idx(i, j + 1), sz_p / dz**2))
                    coeff_center -= sz_p / dz**2

                if neighbors_valid['zm']:
                    sz_m = (sigma[i, j - 1] + sigma[i, j]) / 2
                    stencil.append((idx(i, j - 1), sz_m / dz**2))
                    coeff_center -= sz_m / dz**2

                rows.append(k)
                cols.append(k)
                data.append(coeff_center)
                for col_idx, val in stencil:
                    rows.append(k)
                    cols.append(col_idx)
                    data.append(val)

    # Assemble and solve
    A = sp.coo_matrix((data, (rows, cols)), shape=(Nr * Nz, Nr * Nz)).tocsr()
    psi_vec = spla.spsolve(A, b)
    psi = psi_vec.reshape((Nr, Nz))

    # Compute total flux through z = z_c
    z_c =  int(Nz/2)
    Jz_total = 0.0
    for i in range(Nr):
        if Nz > 1:
            dpsi_dz = (psi[i, z_c+1] - psi[i, z_c]) / dz
        else:
            dpsi_dz = 0
        Jz = -sigma[i, z_c] * dpsi_dz
        Jz_total += 2 * np.pi * r[i] * Jz * dr

    delta_psi = psi_source - psi_sink
    R_total = delta_psi / Jz_total if Jz_total != 0 else np.inf

    print(
        'Total Flux [Φ]:', Jz_total,
        'Total Resistance [R_total]:', R_total)
    
    #The rest of the reservoir before oblate spheroid
    R_left = (np.pi - 2*np.arctan(l1/fields["r"]))/(4*np.pi*fields["r"])/fields["D_0"]
    R_total +=R_left
    
    return R_total, psi.T
#%%
if __name__=="__main__":
    a0 = 0.7
    a1 = -0.3
    L=52
    r_pore=26
    sigma_ = 0.02
    alpha =  30**(1/2)
    d = 20
    chi_PC = -1.3
    chi_PS =0.5

    fields = calculate_fields_in_pore.calculate_fields(
        a0=a0, a1=a1, 
        chi_PC=chi_PC,
        chi_PS=chi_PS,
        wall_thickness = L, 
        pore_radius = r_pore,
        d=d,
        sigma = sigma_,
        mobility_model_kwargs = {"prefactor":alpha},
    )

    R, psi = R_steady_state(fields)

    plt.imshow(psi.T, origin = "lower")
#%%