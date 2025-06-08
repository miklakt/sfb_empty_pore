#%%
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, label, binary_erosion
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator

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
    
def extract_edge(mask):
    dilated = binary_dilation(mask)
    edge = dilated & ~binary_erosion(mask)
    return edge
#%%
def soft_clip_lower(x, min_val, softness=1.0):
    return min_val + softness * np.log1p(np.exp((x - min_val) / softness))

def soft_clip_upper(x, max_val, softness=1.0):
    return max_val - softness * np.log1p(np.exp((x - max_val) / softness))
#%%
def process_infinite_conductivity_clusters(conductivity, bc_source):
    Nr, Nz = np.shape(conductivity)
    mask = np.isinf(conductivity)
    labels, num_clusters = label(mask)
    alias_map = {}
    bc_sink = np.zeros_like(conductivity)

    for cluster_id in range(1, num_clusters + 1):
        cluster = (labels == cluster_id)
        coords = np.argwhere(cluster)
        n_nodes = len(coords)

        touches_source = np.any(bc_source[cluster])
        touches_sink = np.any(coords[:, 1] == Nz - 1)

        if touches_source and touches_sink:
            raise RuntimeError(f"Cluster {cluster_id} with infinite conductivity shorts source to sink.")
        elif touches_source:
            raise RuntimeError(f"Cluster {cluster_id} with infinite conductivity touches source.")
        elif touches_sink:
            print(f"Cluster {cluster_id} with {n_nodes} infinite conductivity nodes touches sink.")
            for i, j in coords:
                bc_sink[i, j] = 1
        else:
            # print(f"Cluster {cluster_id} with {n_nodes} infinite conductivity nodes is floating.")
            # ref_i, ref_j = coords[0]
            # for i, j in coords:
            #     alias_map[(i, j)] = (ref_i, ref_j)
            ValueError("Floating conductors are not implemented")

    return bc_sink#, alias_map


def R_steady_state(conductivity, bc_source):

    def mean(a,b):
        return (a**-1+b**-1)**-1*2

    Nz, Nr = np.shape(conductivity)
    dr, dz = 1,1
    # Grid
    R = np.arange(0, Nr)
    #Z = np.arange(0, Nz)

    bc_sink = process_infinite_conductivity_clusters(conductivity.T, bc_source.T)
    conductivity = np.array(conductivity.T, dtype=np.float64)
    bc_source = np.array(bc_source.T, dtype = np.float64)
    #bc_sink = bc_sink.T

    # Dirichlet boundary conditions
    psi_source = 1.0
    psi_sink = 0.0

    # Helper to map 2D (i,j) to 1D index
    # def idx(i, j):
    #     return alias_map.get((i, j), (i, j))[0] * Nz + alias_map.get((i, j), (i, j))[1]
    def idx(i,j):
        return i*Nz + j

    # Build sparse matrix A for the operator
    data, rows, cols = [], [], []
    b = np.zeros(Nr * Nz)

    def add_stencil_term(i, j, k, coeff_center, stencil):
        rows.append(k)
        cols.append(k)
        data.append(coeff_center)
        for col_idx, val in stencil:
            rows.append(k)
            cols.append(col_idx)
            data.append(val)


    def is_valid(ii, jj):
        return 0 <= ii < Nr and 0 <= jj < Nz and conductivity[ii, jj] != 0

    data, rows, cols = [], [], []
    b = np.zeros(Nr * Nz)

    for i in range(Nr):
        for j in range(Nz):
            k = idx(i, j)
            r_i = R[i]

            if bc_source[i, j]:
                rows.append(k); cols.append(k); data.append(1.0); b[k] = 1.0
                continue
            if bc_sink[i, j]:
                rows.append(k); cols.append(k); data.append(1.0); b[k] = 0.0
                continue

            if j == Nz - 1:
                if conductivity[i, j] != 0:
                    coeff_center = -3.0 * conductivity[i, j] / dz**2
                    coeff_neigh = conductivity[i, j] / dz**2
                    rows.extend([k, k]); cols.extend([k, idx(i, j - 1)]); data.extend([coeff_center, coeff_neigh])
                    b[k] += 2.0 * conductivity[i, j] * 0.0 / dz**2
                else:
                    rows.append(k); cols.append(k); data.append(1.0)
                continue

            if conductivity[i, j] == 0:
                rows.append(k); cols.append(k); data.append(1.0)
                continue

            coeff_center = 0
            stencil = []

            def add_term(ii, jj, coeff):
                nonlocal coeff_center
                stencil.append((idx(ii, jj), coeff))
                coeff_center -= coeff

            if i == 0:
                if is_valid(i + 1, j):
                    kp = idx(i + 1, j)
                    rows.extend([k, k]); cols.extend([k, kp]); data.extend([-1 / dr, 1 / dr])
                if j < Nz - 1 and is_valid(i, j + 1):
                    sz_p = mean(conductivity[i, j], conductivity[i, j + 1])
                    add_term(i, j + 1, sz_p / dz**2)
                if j > 0 and is_valid(i, j - 1):
                    sz_m = mean(conductivity[i, j], conductivity[i, j - 1])
                    add_term(i, j - 1, sz_m / dz**2)
                add_stencil_term(i, j, k, coeff_center, stencil)
            elif i == Nr - 1:
                if is_valid(i - 1, j):
                    km = idx(i - 1, j)
                    rows.extend([k, k]); cols.extend([k, km]); data.extend([-1 / dr, 1 / dr])
                else:
                    rows.append(k); cols.append(k); data.append(1.0)
            else:
                if is_valid(i + 1, j):
                    sr_p = mean(conductivity[i + 1, j], conductivity[i, j])
                    add_term(i + 1, j, sr_p / dr**2 + sr_p / (2 * dr * r_i))
                if is_valid(i - 1, j):
                    sr_m = mean(conductivity[i - 1, j], conductivity[i, j])
                    add_term(i - 1, j, sr_m / dr**2 - sr_m / (2 * dr * r_i))
                if is_valid(i, j + 1):
                    sz_p = mean(conductivity[i, j + 1], conductivity[i, j])
                    add_term(i, j + 1, sz_p / dz**2)
                if is_valid(i, j - 1):
                    sz_m = mean(conductivity[i, j - 1], conductivity[i, j])
                    add_term(i, j - 1, sz_m / dz**2)
                add_stencil_term(i, j, k, coeff_center, stencil)

    data = np.array(data,dtype=np.float64)
    rows = np.array(rows,dtype=np.float64)
    cols = np.array(cols,dtype=np.float64)
    #Assemble and solve
    A = sp.coo_matrix((data, (rows, cols)), shape=(Nr * Nz, Nr * Nz), dtype = np.float64).tocsr()
    psi_vec = spla.spsolve(A, b)

    # A = sp.coo_matrix((data, (rows, cols)), shape=(Nr * Nz, Nr * Nz)).tocsr()
    # ilu = spilu(A.tocsc(), drop_tol=1e-4, fill_factor=10)
    # M = LinearOperator(A.shape, ilu.solve)
    # psi_vec, info = bicgstab(A, b, M=M, maxiter=1000)
    # if info != 0:
    #     raise RuntimeError(f"Solver did not converge. Info = {info}")
    
    psi = psi_vec.reshape((Nr, Nz))

    # # Compute total flux through z = z_c
    # #z_c =  int(Nz/2)
    # z_c = -2
    # Jz_total = 0.0
    # for i in range(Nr):
    #     dpsi_dz = (psi[i, z_c+1] - psi[i, z_c]) / dz
    #     Jz = -mean(conductivity[i, z_c+1],conductivity[i, z_c])* dpsi_dz
    #     Jz_total += np.pi * (2*R[i]) * Jz * dr
    # Compute total flux across full half-cylinder surface (top and side)
    Jz_total = 0.0
    # Axial flux at z = 0 (source face)
    # a quarter cylinder outside the pore
    z_c = int(Nz*0.3+50)
    r_c = int(Nr*0.3+50)
    if z_c + 1 >= Nz or r_c + 1 >= Nr:
        raise ValueError("z_c or r_c exceeds domain bounds.")

    # Axial flux at z = z_c
    for i in range(r_c):
        dpsi_dz = (psi[i, z_c + 1] - psi[i, z_c]) / dz
        cond_face = mean(conductivity[i, z_c], conductivity[i, z_c + 1])
        Jz = -cond_face * dpsi_dz
        Jz_total += np.pi * (2 * R[i]) * Jz * dr

    # Radial flux at r = r_c
    for j in range(z_c, Nz):
        dpsi_dr = (psi[r_c + 1, j] - psi[r_c, j]) / dr
        cond_face = mean(conductivity[r_c + 1, j], conductivity[r_c, j])
        Jr = cond_face * dpsi_dr
        Jz_total += 2 * np.pi * (R[r_c]) * Jr * dz

    delta_psi = psi_source - psi_sink
    R_total = delta_psi / Jz_total if Jz_total != 0 else np.inf

    print(
        'Total Flux [Φ]:', Jz_total,
        'Total Resistance [R_total]:', R_total)
    
    #The rest of the reservoir before oblate spheroid
    # R_left = (np.pi - 2*np.arctan(l1/fields["r"]))/(4*np.pi*fields["r"])/fields["D_0"]
    # R_total +=R_left
    #A = sp.coo_matrix((data, (rows, cols)), shape=(Nr * Nz, Nr * Nz)).tocsr()
    return R_total, psi.T, A



#%%

def pad_fields(fields, z_boundary):
    conductivity = fields["conductivity"][:]
    walls = fields["walls"]
    r_pore = fields["r"]
    d = fields["d"]
    # we ake only the z minus side, the problem is symmetric
    conductivity = conductivity[:np.shape(conductivity)[0]//2]
    walls = walls[:np.shape(walls)[0]//2]

    bulk_conductivity = fields["conductivity"][1,1]
    l1 = fields["l1"]
    pad_z = z_boundary-l1+1
    if pad_z>0:
        conductivity = np.pad(conductivity, ((pad_z,0),(0,0)), "constant", constant_values=bulk_conductivity)
        walls = np.pad(walls, ((pad_z,0),(0,0)), "edge")

    # r=np.shape(conductivity)[1]
    major_axis = int(np.sqrt(z_boundary**2 + r_pore**2/2))
    pad_r = major_axis-np.shape(conductivity)[1]+1

    if pad_r>0:
        conductivity = np.pad(conductivity, ((0,0),(0,pad_r)), "constant", constant_values=bulk_conductivity)
        walls = np.pad(walls, ((0,0),(0,pad_r)), "edge")
    
    conductivity[walls==True]=0.0
    #conductivity[-1] = 0.0

    z,r = np.shape(conductivity)
    R=np.arange(0,r)
    Z=np.arange(0,z)
    RR,ZZ=np.meshgrid(R,Z)
    x0 = z_boundary+1
    y0 = 0
    bc_source = ~is_inside_ellipse(ZZ,RR,a=int(r_pore-d/2), b=z_boundary, x0=x0, y0=y0, side = "left")
    bc_source[z_boundary:] = False
    bc_source[walls==True] = False

    return conductivity, bc_source

def R_solve(fields, z_boundary = 1000, conductivity_min=1e-8, conductivity_max = 1e8, clip=True):
    conductivity, bc_source = pad_fields(fields, z_boundary)
    if np.nanmax(conductivity)>=conductivity_max:print("Very high conductivity")
    #if np.nanmin(conductivity)<=1e-10:print("Very low conductivity")
    if clip:
        conductivity[conductivity>=conductivity_max]=conductivity_max
        conductivity[conductivity<=conductivity_min]=conductivity_min
    else:
        conductivity[conductivity>=conductivity_max]=np.inf
        conductivity[conductivity<=conductivity_min]=0
    R, psi, A = R_steady_state(conductivity, bc_source)
    #psi = psi[:np.shape(psi)[0]-1]
    psi = psi[:np.shape(psi)[0]]

    #bc_source = bc_source[:np.shape(psi)[0]-1]
    bc_source = bc_source[:np.shape(psi)[0]]
    bc_source = np.concatenate([bc_source, np.zeros_like(bc_source)], axis = 0)

    conductivity = conductivity[:np.shape(psi)[0]]
    conductivity = np.concatenate([conductivity, conductivity[::-1]], axis = 0)

    psi_mirror = np.flip(psi, axis=0)
    psi = 0.5+psi/2
    psi_mirror = 0.5-psi_mirror/2
    psi = np.concatenate([psi, psi_mirror], axis=0)
    grad_x, grad_y = np.gradient(psi)
    grad_x*=conductivity
    grad_y*=conductivity

    x,y=fields["xlayers"],fields["ylayers"]
    y_,x_ = np.shape(psi)
    crop_y = (y_-y)//2
    crop_x = x_-x


    walls = fields["walls"]
    walls = np.pad(walls, ((crop_y,crop_y),(0,crop_x)), "edge")
    fe = np.pad(fields["free_energy"], ((crop_y,crop_y),(0,crop_x)), "constant", constant_values = 0)

    psi[walls==True] = np.nan
    c =  psi*np.exp(-fe)

    structure = np.array([[1,1,1]])
    walls_y = binary_dilation(walls, structure=structure)
    walls_x = binary_dilation(walls, structure=structure.T)
    grad_x[walls_x==True] = 0
    grad_y[walls_y==True] = 0

    #The rest of the reservoir before oblate spheroid
    r_pore = fields["r"]
    d = fields["d"]
    R_left = (np.pi - 2*np.arctan(z_boundary/(r_pore-d/2)))/(4*np.pi*(r_pore-d/2))/fields["D_0"]
    R +=R_left
    R *=2


    nocrop_fields = {
        "psi":psi,
        "walls":walls,
        "J_z":grad_x,
        "J_r":grad_y,
        "c": c,
        "s" : bc_source,
        "conductivity"  : conductivity
    }

    psi = psi[crop_y:-crop_y,:x]
    #grad_x = grad_x[crop_y:-crop_y,:x]
    #grad_y = grad_y[crop_y:-crop_y,:x]
    c = c[crop_y:-crop_y,:x]

    l1= fields["l1"]
    s = fields["s"]
    #not exactly correct because the defined on staggered lattice
    a_z = np.arange(0, int(r_pore))
    a_z[0] += 1/4
    a_z[-1]-= 1/4
    pore_conductivity = conductivity[int(z_boundary-d/2+1):int(z_boundary+s+d/2+1),:int(r_pore)]
    R_int = np.sum((np.pi*np.sum(pore_conductivity*(2*a_z), axis = 1))**(-1))
    R_ext = R - R_int

    if not np.isfinite(R_int):
        print("Infinite resistance")
        R = np.inf
        R_int = np.inf
        R_ext = np.nan

    fields["psi"] = psi
    fields["R_lin_alg"] = R
    fields["R_lin_alg_int"] = R_int
    fields["R_lin_alg_ext"] = R_ext
    #fields["J_z"] = grad_x
    #fields["J_r"] = grad_y
    fields["c"] = c
    return nocrop_fields, A

def R_empty_pore(pore_radius:int, wall_thickness:int, d:int = None, z_boundary = 1000):
    if d is None:
        D_0 = 1.0
    else:
        if d>1:
            if d//2 != d/2:
                raise ValueError("d has to be even")
            if d>=2*pore_radius:
                raise ValueError("d > pore_radius")
        D_0 = 1/(3*np.pi*d)

    from calculate_fields_in_pore import add_walls

    major_axis = int(np.sqrt(z_boundary**2 + pore_radius**2/2))

    xlayers = int(major_axis+1)
    ylayers = z_boundary*2+wall_thickness+2

    conductivity = np.ones((ylayers, xlayers))*D_0
    walls = np.zeros_like(conductivity)
    walls[z_boundary+1:z_boundary+wall_thickness+1, pore_radius:] = True
    def generate_circle_kernel(d):
        radius = d/2
        a = np.zeros((d, d), dtype =bool)
        radius2 = radius**2
        for i in range(d):
            for j in range(d):
                distance2 = (radius-i-0.5)**2 + (radius-j-0.5)**2
                if distance2<radius2:
                    a[i,j] = True
        return a
    if (d is not None) and (d>1):
        walls = binary_dilation(walls, structure=generate_circle_kernel(d))
    conductivity[walls==1] = 0.0

    conductivity = conductivity[:np.shape(conductivity)[0]//2]
    walls = walls[:np.shape(walls)[0]//2]
    
    R=np.arange(0,xlayers)
    Z=np.arange(0,ylayers//2)
    RR,ZZ=np.meshgrid(R,Z)
    x0 = z_boundary+1
    y0 = 0
    bc_source = ~is_inside_ellipse(ZZ,RR,a=int(pore_radius-d/2), b=z_boundary, x0=x0, y0=y0, side = "left")
    bc_source[z_boundary:] = False
    bc_source[walls==True] = False
    
    R_tot, psi, A = R_steady_state(conductivity, bc_source)
    # psi = psi[:np.shape(psi)[0]-1]

    # bc_source = bc_source[:np.shape(psi)[0]-1]
    # bc_source = np.concatenate([bc_source, np.zeros_like(bc_source)], axis = 0)

    # psi_mirror = np.flip(psi, axis=0)
    # psi = 0.5+psi/2
    # psi_mirror = 0.5-psi_mirror/2
    # psi = np.concatenate([psi, psi_mirror], axis=0)
    # grad_x, grad_y = np.gradient(psi, edge_order=2)

    R_left = (np.pi - 2*np.arctan(z_boundary/pore_radius))/(4*np.pi*pore_radius)/D_0
    R_tot+=R_left
    #R*=2

    a_z = np.arange(0, int(pore_radius))
    a_z[0]+=1/4
    a_z[-1]-=1/4
    pore_conductivity = conductivity[int(z_boundary+1-d/2):,:int(pore_radius)]
    R_int = np.sum((np.pi*np.sum(pore_conductivity*(2*a_z), axis = 1))**(-1))
    R_ext = R_tot - R_int

    fields = {}
    #fields["psi"] = psi
    fields["R"] = R_tot*2
    fields["R_int"] = R_int*2
    fields["R_ext"] = R_ext*2
    #fields["J_z"] = grad_y
    #fields["J_r"] = grad_x
    return fields

#%%
if __name__=="__main__":
    import calculate_fields_in_pore
    from matplotlib import patches as mpatches
    from matplotlib import rc
    rc('hatch', color='darkgreen', linewidth=9)
    a0 = 0.7
    a1 = -0.3
    L = 52
    r_pore=26
    sigma = 0.02
    alpha =  30**(1/2)
    d = 30
    chi_PC = -1.8
    chi_PS =0.5

    fields = calculate_fields_in_pore.calculate_fields(
        a0=a0, a1=a1, 
        chi_PC=chi_PC,
        chi_PS=chi_PS,
        wall_thickness = L, 
        pore_radius = r_pore,
        d=d,
        sigma = sigma,
        mobility_model_kwargs = {"prefactor":alpha},
        linalg=False,
        #gel_phi=0.3
    )
    
    nocrop_fields, A = R_solve(fields, z_boundary=250)
    x, y = np.shape(nocrop_fields["psi"])
    extent = [-x/2, x/2, 0, y]

    fig, ax = plt.subplots()
    
    bg = mpatches.Rectangle(
        (0, 0), 1, 1,               # (x, y), width, height in axes coordinates
        transform=ax.transAxes,    # makes it relative to axes (0-1 range)
        facecolor='green',          # transparent fill
        edgecolor='darkgreen',         # hatch color
        hatch='/',               # hatch pattern
        zorder=-10                 # draw below everything else
    )
    ax.add_patch(bg)
    #ax.imshow(conductivity.T, interpolation="none", origin = "lower")
    im = ax.imshow(
        nocrop_fields["psi"].T, 
        interpolation="none", 
        origin = "lower",
        extent = extent
        )
    s = nocrop_fields["s"].astype(float)
    s[s==0] = np.nan
    #s[133:143,133:143] = 100

    ax.imshow(
        s.T, 
        interpolation="none", 
        origin = "lower",
        #alpha=0.5,
        cmap = "Greens_r",
        extent = extent
        )
    
    ax.imshow(
        s[::-1].T, 
        interpolation="none", 
        origin = "lower",
        #alpha=0.5,
        cmap = "Blues_r",
        extent = extent
        )
    
    # cs = ax.contour(nocrop_fields["c"].T, 
    #     interpolation="none", 
    #     origin = "lower",
    #     colors = "k",
    #     levels = [0.999, 0.99, 0.9 ,0.75, 0.5, 0.25, 0.1, 0.01, 0.001][::-1],
    #     extent = extent
    #     )

    # ax.set_xlim(-50,50)
    # ax.set_ylim(990)
    #ax.clabel(cs, cs.levels)

    ax.set_xlabel("$z$")
    ax.set_ylabel("$r$")

    ax.set_aspect("equal")
    cbar = plt.colorbar(im)
    #%%
    fig, ax = plt.subplots()
    J_z = np.sum(-np.pi*nocrop_fields["J_z"]*(2*np.arange(y)+1), axis = 1)
    z = np.arange(x)-extent[1]
    ax.plot(z,J_z)
    #%%
    fig, ax = plt.subplots()
    R_z = np.sum(np.pi*nocrop_fields["conductivity"]*(2*np.arange(y)+1), axis = 1)**-1
    z = np.arange(x)-extent[1]
    ax.plot(z,R_z)
# %%
    psi = nocrop_fields["psi"]
    conductivity = nocrop_fields["conductivity"]
    #grad_r = psi[:,1:] - psi[:,:-1]
    #grad_z = psi[1:, :] - psi[:-1, :]
    grad_z, grad_r = np.gradient(psi, edge_order=1)
    grad_z*=conductivity
    grad_r*=conductivity
    #div_psi = np.sum(np.gradient(psi, edge_order=1),axis=0)

    im = plt.imshow(conductivity.T, origin = "lower", interpolation = "none",
               extent = extent
               )
    #plt.xlim(-30,30)
    #plt.ylim(0,30)
    #plt.axvline(277, color = "red")
    plt.colorbar(im)
# %%
    plt.plot(grad_z[:,0])

# %%
    plt.plot(grad_z[250,:])
# %%
