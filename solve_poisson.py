#%%
import typing
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, label, binary_erosion
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator, gmres


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

FieldType = np.typing.NDArray[np.float64]
class PoissonSolver2DCylindrical:
    def __init__(self, D : FieldType, S : FieldType, BC = None, orientation = "rz", dr = 1, dz = 1):
        self.D = D
        # if isinstance(S, FieldType):
        #     self.S = {"":S}
        # else:
        self.S = S
        self.BC = BC
        self.Nr, self.Nz = np.shape(self.D)
        self.N = self.Nr*self.Nz
        self.dr = dr
        self.dz = dz
        self.b = np.zeros(self.Nr*self.Nz)
    def idx_ij(self, k:int):
        if k>=self.N:
            raise IndexError(f"flatten list index {k} out of range {self.N}")
        return k//self.Nz, k%self.Nz
    def idx_k(self, i:int, j:int):
        if i>= self.Nr:
            raise IndexError(f"list index {i=} out of range {self.Nr}")
        if j>= self.Nz:
            raise IndexError(f"list index {j=} out of range {self.Nz}")
        k = i*self.Nz + j
        return k
    def idx(self, i:int=None, j:int=None, k:int =None):
        """Complete ij and flatten indices

        Args:
            i (int, optional): _description_. Defaults to None.
            j (int, optional): _description_. Defaults to None.
            k (int, optional): _description_. Defaults to None.

        Raises:
            ValueError: either ij indices or k index has to be provided 

        Returns:
            _type_: ijk
        """
        if k is not None:
            if (i is None) and (j is None):
                return self.idx_ij(k), k
            else:
                raise ValueError("too many indices provided")
        else:
            return i, j, self.idx_k(i,j)
    def get_position(self, i:int=None, j:int=None, k:int=None):
        # returns the left lower corner of the cell
        # the physical qt are sampled at z+0.5dz r+0.5dr offset
        i, j, k = self.idx(i,j,k)
        r = i*self.dr
        z = j*self.dz
        return r, z
    def get_cell_volume(self,  i:int=None, j:int=None, k:int=None):
        r, z = self.get_position(i, j, k)
        volume = (2*r+1)*self.dr*self.dz
        return volume
    def get_faces(self, i:int=None, j:int=None, k:int=None):
        r, z = self.get_position(i, j, k)
        rm = 2*r*self.dz
        rp = 2*(r+self.dr)*self.dz
        zm = zp = 2*r*self.dr
        return {"rm":rm, "rp":rp, "zm":zm, "zp":zp}
    def get_lambdas(self, i:int=None, j:int=None, k:int=None):
        r, z = self.get_position(i, j, k)
        rm = 2*r/(2*r+1) / self.dr**2
        rp= (2*r+2*self.dr)/(2*r+1) / self.dr**2
        zm = zp = 1/self.dz**2
        return {"rm":rm, "rp":rp, "zm":zm, "zp":zp}
    
    def get_D_faces(self, i:int=None, j:int=None, k:int=None):
        def mean(a,b):
            if (a==0) or (b==0): return 0.0
            return 2/(a**-1+b**-1)
        #by default we set Neumann zero-flux conditions
        if i==0:
            rm = 0
        else:
            rm =  mean(self.D[i - 1, j],self.D[i, j])
        if i==self.Nr-1:
            rp = 0
        else:
            rp =  mean(self.D[i + 1, j],self.D[i, j])
        if j == 0:
            zm = 0
        else:
            zm =  mean(self.D[i, j - 1],self.D[i, j])
        if j == self.Nz-1:
            zp = 0
        else:
            zp =  mean(self.D[i, j + 1],self.D[i, j])
        return {"rm":rm, "rp":rp, "zm":zm, "zp":zp}
    
    def get_stencil(self, i:int=None, j:int=None, k:int=None):
        i,j,k = self.idx(i,j,k)
        stencil = {"rm":0, "rp":0, "zm":0, "zp":0, "c":0}
        l = self.get_lambdas(i,j)
        D = self.get_D_faces(i,j)
        source = self.S
        source_val = 1.0
        sink_val = 0.0

        r, z = self.get_position(i,j)
        dr = self.dr
        dz = self.dz

        b = self.b

        print(i,j,k)
        print(D)
        if source[i,j]:
            print("source")
            stencil["c"] = source_val
            b[k] = source_val
        if self.D[i,j] == 0:
            print("D=0")
            stencil["c"] = 0.0
        else:
            # This should not be hard-codded,
            # but handled by choosing BC
            # now it is Dirichlet on the outermost face 
            if j==self.Nz-1:
                print("last")
                stencil["zm"]+= self.D[i,j]*l["zm"]*2.0
                stencil["c"]-= self.D[i,j]*l["zm"]*3.0
                b[k] = self.D[i,j]*l["zm"]*2.0*sink_val
                #block radial flux
                D["rm"] =  D["rp"] = 0
                #block already edited
                D["zm"] = 0

            coef = D["rm"]*l["rm"]
            stencil["rm"]+=coef
            stencil["c"]-=coef

            coef = D["rp"]*l["rp"]
            stencil["rp"]+=coef
            stencil["c"]-=coef

            coef = D["zm"]*l["zm"]
            stencil["zm"]+=coef
            stencil["c"]-=coef
            
            coef = D["zp"]*l["zp"]
            stencil["zp"]+=coef
            stencil["c"]-=coef

            
        return stencil
    
    def build_matrix(self):
        data, rows, cols = [], [], []
        Nr, Nz = self.Nr, self.Nz

        for i in range(Nr):
            for j in range(Nz):
                k = self.idx_k(i, j)

                # Wall cell: D = 0 → enforce φ = 0 (or keep φ undefined)
                if self.D[i, j] == 0:
                    rows.append(k)
                    cols.append(k)
                    data.append(1.0)
                    self.b[k] = 0.0
                    continue  # Skip stencil construction

                stencil = self.get_stencil(i, j)

                # Center
                rows.append(k)
                cols.append(k)
                data.append(stencil["c"])

                # Neighbors, only if D ≠ 0 at neighbor
                if stencil["rm"] != 0 and i > 0 and self.D[i - 1, j] != 0:
                    rows.append(k)
                    cols.append(self.idx_k(i - 1, j))
                    data.append(stencil["rm"])

                if stencil["rp"] != 0 and i < Nr - 1 and self.D[i + 1, j] != 0:
                    rows.append(k)
                    cols.append(self.idx_k(i + 1, j))
                    data.append(stencil["rp"])

                if stencil["zm"] != 0 and j > 0 and self.D[i, j - 1] != 0:
                    rows.append(k)
                    cols.append(self.idx_k(i, j - 1))
                    data.append(stencil["zm"])

                if stencil["zp"] != 0 and j < Nz - 1 and self.D[i, j + 1] != 0:
                    rows.append(k)
                    cols.append(self.idx_k(i, j + 1))
                    data.append(stencil["zp"])

        A = sp.coo_matrix((data, (rows, cols)), shape=(self.N, self.N)).tocsr()
        return A, self.b
    
    def compute_flux_faces_conservative(self, psi: FieldType):
        Nr, Nz = self.Nr, self.Nz
        dr, dz = self.dr, self.dz
        D = self.D

        J_faces = {
            "J_rm": np.zeros_like(psi),
            "J_rp": np.zeros_like(psi),
            "J_zm": np.zeros_like(psi),
            "J_zp": np.zeros_like(psi),
        }

        for i in range(Nr):
            for j in range(Nz):
                if D[i, j] == 0:
                    continue  # wall: no flux

                # --- r− face ---
                if i > 0 and D[i - 1, j] != 0:
                    D_face = 0.5 * (D[i, j] + D[i - 1, j])
                    dpsi = (psi[i, j] - psi[i - 1, j]) / dr
                    J_faces["J_rm"][i, j] = -D_face * dpsi

                # --- r+ face ---
                if i < Nr - 1 and D[i + 1, j] != 0:
                    D_face = 0.5 * (D[i, j] + D[i + 1, j])
                    dpsi = (psi[i + 1, j] - psi[i, j]) / dr
                    J_faces["J_rp"][i, j] = -D_face * dpsi

                # --- z− face ---
                if j > 0 and D[i, j - 1] != 0:
                    D_face = 0.5 * (D[i, j] + D[i, j - 1])
                    dpsi = (psi[i, j] - psi[i, j - 1]) / dz
                    J_faces["J_zm"][i, j] = -D_face * dpsi

                # --- z+ face ---
                if j < Nz - 1 and D[i, j + 1] != 0:
                    D_face = 0.5 * (D[i, j] + D[i, j + 1])
                    dpsi = (psi[i, j + 1] - psi[i, j]) / dz
                    J_faces["J_zp"][i, j] = -D_face * dpsi

        return J_faces


    
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
    d = 12
    chi_PC = -1.5
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
    #%%
    conductivity, source = pad_fields(fields, z_boundary=300)
    # %%
    poisson = PoissonSolver2DCylindrical(D=conductivity.T, S=source.T)
    #%%
    A, b = poisson.build_matrix()
    #%%
    psi_vec = spla.spsolve(A, b)
    psi = psi_vec.reshape((poisson.Nr,poisson.Nz))
    J = poisson.compute_flux_faces_conservative(psi)
    #%%
    D = np.ones((10,20))
    D[2:,-2:] = 0
    S = np.zeros_like(D)
    S[:,0]=1.0
    poisson = PoissonSolver2DCylindrical(D, S)
    A, b = poisson.build_matrix()
    psi_vec = spla.spsolve(A, b)
    psi = psi_vec.reshape((poisson.Nr,poisson.Nz))

    
# %%
