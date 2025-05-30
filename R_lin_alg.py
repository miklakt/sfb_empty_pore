#%%
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

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

#%%
def R_steady_state(conductivity, bc_source):
    # l1 = fields["l1"]
    # pore_radius = fields["r"]
    # pad_sides = 100
    # pad_top = np.sqrt((l1)**2 + (pore_radius / 2)**2)
    #pad_top = np.sqrt(4*(l1+pad_sides)**2 + pore_radius**2)
    #pad_fields(fields, pad_sides, pad_top)

    Nz, Nr = np.shape(conductivity)
    dr, dz = 1,1
    # Grid
    R = np.arange(0, Nr)
    Z = np.arange(0, Nz)
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')

    sigma = conductivity[ZZ, RR]
    R_local =sigma**-1

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
            r_i = R[i]

            if bc_source[j, i]:
                rows.append(k)
                cols.append(k)
                data.append(1.0)
                b[k] = psi_source
                continue
            elif j==Nz-1:
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
        Jz_total += 2 * np.pi * R[i] * Jz * dr

    delta_psi = psi_source - psi_sink
    R_total = delta_psi / Jz_total if Jz_total != 0 else np.inf

    print(
        'Total Flux [Φ]:', Jz_total,
        'Total Resistance [R_total]:', R_total)
    
    #The rest of the reservoir before oblate spheroid
    # R_left = (np.pi - 2*np.arctan(l1/fields["r"]))/(4*np.pi*fields["r"])/fields["D_0"]
    # R_total +=R_left
    
    return R_total, psi.T
#%%

def pad_fields(fields, z_boundary):
    conductivity = fields["conductivity"]
    walls = fields["walls"]
    r_pore = fields["r"]
    #we ake only the left side, the problem is symmetric
    conductivity = conductivity[:np.shape(conductivity)[0]//2+1]
    walls = walls[:np.shape(walls)[0]//2+1]

    bulk_conductivity = fields["conductivity"][1,1]
    l1 = fields["l1"]
    pad_z = z_boundary-l1+1
    if pad_z>0:
        conductivity = np.pad(conductivity, ((pad_z,0),(0,0)), "constant", constant_values=bulk_conductivity)
        walls = np.pad(walls, ((pad_z,0),(0,0)), "edge")

    r=np.shape(conductivity)[1]
 
    major_axis = int(np.sqrt(z_boundary**2 + r_pore**2/2))
    pad_r = major_axis-np.shape(conductivity)[1]+1

    if pad_r>0:
        bulk = fields["conductivity"][1,1]
        conductivity = np.pad(conductivity, ((0,0),(0,pad_r+1)), "constant", constant_values=bulk)
        walls = np.pad(walls, ((0,0),(0,pad_r+1)), "edge")
    
    conductivity[walls==True]=0.0

    z,r = np.shape(conductivity)
    R=np.arange(0,r)
    Z=np.arange(0,z)
    RR,ZZ=np.meshgrid(R,Z)
    x0 = z_boundary
    y0 = 0
    bc_source = ~is_inside_ellipse(ZZ,RR,a=r_pore, b=z_boundary, x0=x0, y0=0, side = "left")
    bc_source[z_boundary:] = False
    bc_source[walls==True]=False

    return conductivity, bc_source

def R_solve(fields, z_boundary = 200):
    conductivity, bc_source = pad_fields(fields, z_boundary)
    R, psi = R_steady_state(conductivity, bc_source)
    psi = psi[:np.shape(psi)[0]-1]

    
    psi_mirror = np.flip(psi, axis=0)
    psi = 0.5+psi/2
    psi_mirror = 0.5-psi_mirror/2
    psi = np.concatenate([psi, psi_mirror], axis=0)
    grad_x, grad_y = np.gradient(psi, edge_order=2)

    x,y=fields["xlayers"],fields["ylayers"]
    y_,x_ = np.shape(psi)
    crop_y = (y_-y)//2
    psi = psi[crop_y:-crop_y,:x]

    grad_x = grad_x[crop_y:-crop_y,:x]
    grad_y = grad_y[crop_y:-crop_y,:x]


    walls = fields["walls"]
    psi[walls==True] = np.nan

    structure = np.array([[1,1,1]])
    walls_y = binary_dilation(walls, structure=structure)
    walls_x = binary_dilation(walls, structure=structure.T)
    grad_x[walls_x==True] = 0
    grad_y[walls_y==True] = 0

    fields["psi"] = psi
    
    #The rest of the reservoir before oblate spheroid
    R_left = (np.pi - 2*np.arctan(z_boundary/fields["r"]))/(4*np.pi*fields["r"])/fields["D_0"]
    R +=R_left

    fields["R_lin_alg"] = R*2
    fields["J_z"] = grad_y
    fields["J_r"] = grad_x
    fields["c"] = fields["psi"]*np.exp(-fields["free_energy"])
    

if __name__=="__main__":
    import calculate_fields_in_pore
    a0 = 0.7
    a1 = -0.3
    L=52
    r_pore=26
    sigma_ = 0.02
    alpha =  30**(1/2)
    d = 12
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
        linalg=False
    )
    
    R_solve(fields)

    

    fig, ax = plt.subplots()
    #ax.imshow(conductivity.T, interpolation="none", origin = "lower")
    ax.imshow(fields["psi"].T, interpolation="none", origin = "lower")
    # ax.scatter([x0],[y0],marker="x",color = "red")
    # ax.scatter([x0-z_boundary],[y0],marker="x",color = "red")
    # c=int(np.sqrt(z_boundary**2 + r_pore**2/2))
    # ax.scatter([x0],[c],marker="x",color = "red")

    ax.set_xlabel("$z$")
    ax.set_ylabel("$r$")

    ax.set_aspect("equal")
#%%