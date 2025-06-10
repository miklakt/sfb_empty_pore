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
    def __init__(self, D : FieldType, S : FieldType | typing.Dict[FieldType], BC = None, orientation = "rz", dr = 1, dz = 1):
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
    # def get_lambdas(self, i:int=None, j:int=None, k:int=None):
    #     v =
        
    #def get_neigh
    # def get_cell_type(self, i:int=None, j:int=None, k:int=None):
    #     i, j, k = self.idx(i,j,k)
    #     source = self.S
    #     if source[i,j]: return "source"
    #     if i==0: return "rm"
    #     if i==self.Nr-1: return "rp"
    #def add_terms(self, stencil)
    def get_stencil(self, i:int=None, j:int=None, k:int=None):
        i,j,k = self.idx(i,j,k)
        stencil = {"rm":0, "rp":0, "zm":0, "zp":0, "c":0}
        source = self.S
        source_val = 1.0

        r, z = self.get_position(i,j)
        dr = self.dr
        dz = self.dz

        if source[i,j]:
            stencil["c"] = source_val
        if i==0:
            Drm =  (self.D[i - 1, j] + self.D[i, j])/2
            lambda_rm = 2*r/(2*r*dr + dr**2)
            stencil["rm"]+=Drm * lambda_rm
            stencil["c"]-=Drm * lambda_rm

            Drp =  (self.D[i + 1, j] + self.D[i, j])/2
            lambda_rm = (2*r+dr)/(2*r*dr + dr**2)
            stencil["rm"]+=Drp * lambda_rm
            stencil["c"]-=Drp * lambda_rm
            

        
        

    




#%%
