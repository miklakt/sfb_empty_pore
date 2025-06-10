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
            if (a==0) or (b==0): return b
            return (a+b)/2
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
        if source[i,j]:
            print("source")
            stencil["c"] = source_val
            b[k] = source_val
        else:
            # This should not be hard-codded,
            # but handled by choosing BC
            # now it is Dirichlet on the outermost face 
            if j==self.Nz-1:
                print("last")
                coef = self.D[i,j]*l["zm"]*2.0
                stencil["zm"]+=coef
                stencil["c"]-=coef
                b[k] = coef*sink_val
                #block radial flux
                D["rm"] =  D["rp"] = 0

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
    
    def get_stencil_indices(self, i:int=None, j:int=None, k:int=None):
        i, j, k = self.idx(i,j,k)
        stencil_indices = {"rm":(i-1,j), "rp":(i-1,j), "zm":(i,j-1), "zp":(i,j+1), "c":(i,j)}

    
    def build_matrix(self):
        data, rows, cols = [], [], []
        b = self.b
        Nr = self.Nr
        Nz = self.Nz
        
        for i in range(Nr):
            for j in range(Nz):
                k = self.idx_k(i,j)
                for neigh, coeff

#%%
D = np.ones((10,10))
S = np.zeros_like(D)
S[:,:3] = True

poisson = PoissonSolver2DCylindrical(D,S)
# %%
poisson.get_stencil(2,9)
# %%
