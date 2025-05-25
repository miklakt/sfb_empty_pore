#%%
import os, sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from calculate_fields_in_pore import *
import pystencils as ps
import sympy as sp
import h5py
import tqdm
import numpy as np
import time

__cupy__ = True
if __cupy__:
    import cupy as xp
else:
    import numpy as xp
#%%
d1facex = lambda x: xp.pad((x[1:,:] + x[:-1,:])/2, ((0,1),(0,0)), "edge")
d1facey = lambda x: xp.pad((x[:,1:] + x[:,:-1])/2, ((0,0),(0,1)), "edge")
d1diffx = lambda x: xp.pad((x[1:,:] - x[:-1,:]), ((0,1),(0,0)), "edge")
d1diffy = lambda x: xp.pad((x[:,1:] - x[:,:-1]), ((0,0),(0,1)), "edge")

def alpha_power_law(Pe):
    alpha = (xp.exp(Pe/2)-1)/(xp.exp(Pe)-1)
    alpha[xp.isclose(Pe, 0)]=0.5
    return alpha


def sigmoid_k_one_over_k(steepness, k, x):
    return (k-1/k)/(1+xp.exp(steepness*x+xp.log(k))) + 1/k

class DriftDiffusionKernelFactory:   
    def __initialize_arrays(self, domain_size):
        
        # radial coordinate of the cell in the cylindrical coordinates
        self.r_arr = xp.ones(domain_size)
        self.r_arr[:]= xp.arange(0, self.rlayers)
        # west-east cells contact area
        self.a_we =  2*self.r_arr
        self.a_we[:,0] = 1/4
        
        # the volume and contact area depends on radial coordinates 
        # lambda_i accounts for the change of contact area relative to the volume
        # going north in cylindrical coordinate
        self.lambda_n_arr = 1+1/(self.r_arr*2)
        self.lambda_n_arr[:, 0] = 2

        # going south in cylindrical coordinates
        self.lambda_s_arr = 1-1/(self.r_arr*2)
        self.lambda_s_arr[:, 0] = 0

        # no walls by default
        self.W_arr = xp.zeros(domain_size, dtype="int8")

        # constant diffusion coefficient by default
        self.D_arr = xp.ones(domain_size)
        
        # no potential by default
        self.U_arr = xp.zeros(domain_size)

        #++updating fields++
        self.c_arr = xp.zeros(domain_size)

        self.J_arr = xp.zeros((*domain_size,2))
        self.J_dif_arr = xp.zeros((*domain_size,2))
        self.J_adv_arr = xp.zeros((*domain_size,2))
        self.grad_c_arr = xp.zeros((*domain_size,2))

        self.div_J_arr = xp.zeros((domain_size))
        # self.div_J_prev_arr = xp.zeros((domain_size))

    def __set_arrays(self, kwargs):
        for kwarg, value in kwargs.items():
            try:
                if isinstance(value, xp.ndarray):
                    xp.copyto(getattr(self, kwarg), value)
                else:
                    raise TypeError("Wrong type, use numpy.ndarray")
            except:
                print(f"object has no attribute {kwarg}")

    @staticmethod
    def __redefine_walls_to_faces(W_arr):
        W_arr = xp.pad((W_arr[1:,:] + W_arr[:-1,:]), ((0,1),(0,0)), "edge")
        W_arr[(W_arr>0)]=1
        return W_arr

    def __get_gradients_and_face_values(self):
        #++diffusion values at the faces++
        #if D_arr is not None: xp.copyto(self.D_arr, D_arr)
        self.D_x_arr = d1facex(self.D_arr)*self.W_not_arr
        self.D_y_arr = d1facey(self.D_arr)*self.W_not_arr

        #++potential gradient values at the faces++
        #if U_arr is not None: xp.copyto(self.U_arr, U_arr)
        self.dU_x_arr = d1diffx(self.U_arr)*self.W_not_arr
        self.dU_y_arr = d1diffy(self.U_arr)*self.W_not_arr

    def __init_fields(self):
        #++constant fields++
        self.W = ps.fields("W: [2d]", W = self.W_arr)
        self.D_x, self.D_y = ps.fields(
            "D_x, D_y: [2d]", 
            D_x = self.D_x_arr, 
            D_y = self.D_y_arr
            )
        self.dU_x, self.dU_y = ps.fields(
            "dU_x, dU_y: [2d]", 
            dU_x = self.dU_x_arr, 
            dU_y = self.dU_y_arr
            )
        self.Pe_x, self.Pe_y = ps.fields(
            "Pe_x, Pe_y: [2d]", 
            Pe_x = self.Pe_x_arr, 
            Pe_y = self.Pe_y_arr
            )

        self.alpha_x, self.alpha_y = ps.fields(
            "alpha_x, alpha_y: [2d]", 
            alpha_x = self.alpha_x_arr,
            alpha_y = self.alpha_y_arr
            )

        self.lambda_n, self.lambda_s = ps.fields(
            "lambda_n, lambda_s: [2d]", 
            lambda_n = self.lambda_n_arr, 
            lambda_s = self.lambda_s_arr
            )
        


        #++updating fields++
        self.c, self.c_next = ps.fields(
            "c, c_next: [2d]", 
            c=self.c_arr, 
            c_next=self.c_arr
            )
        
        self.J = ps.fields("J(2): [2d]",
                           J = self.J_arr)

        self.J_dif = ps.fields("J_dif(2): [2d]",
                    J_dif = self.J_dif_arr)
        
        self.J_adv = ps.fields("J_adv(2): [2d]",
                J_adv = self.J_adv_arr)
    

        
        self.grad_c = ps.fields("grad_c(2): [2d]",
                           grad_c = self.grad_c_arr)
                
        self.div_J = ps.fields("div_J: [2d]",
                           div_J = self.div_J_arr)
        
        # self.div_J_prev = ps.fields("div_J_prev: [2d]", 
        #                    div_J_prev = self.div_J_prev_arr)

    def __init__(self, W_arr = None, U_arr = None, D_arr = None, differencing = "hybrid", **kwargs) -> None:
        if W_arr is not None:
            W_arr=DriftDiffusionKernelFactory.__redefine_walls_to_faces(xp.array(W_arr))

        for arr in W_arr, U_arr, D_arr:
            try:
                self.zlayers, self.rlayers = xp.shape(arr)
            except ValueError:
                pass
        try:
            domain_size = (self.zlayers, self.rlayers)
        except AttributeError:
            self.zlayers = kwargs["zlayers"]
            self.rlayers = kwargs["rlayers"]
            domain_size = (self.zlayers, self.rlayers)

        self.__initialize_arrays(domain_size)
        
        self.__set_arrays(dict(W_arr=W_arr, U_arr=U_arr, D_arr=D_arr))
        self.__set_arrays(kwargs)
        
        self.W_not_arr=(self.W_arr==0)

        self.__get_gradients_and_face_values()

        #++Peclet number at the faces++
        # TODO change for variable dx
        self.Pe_x_arr = -self.dU_x_arr
        self.Pe_y_arr = -self.dU_y_arr

        self.alpha_x_arr = xp.ones(domain_size)*1/2
        self.alpha_y_arr = xp.ones(domain_size)*1/2

        #Hybrid differencing
        if differencing == "hybrid":
            self.alpha_x_arr[(self.Pe_x_arr >= 2)]=0.0
            self.alpha_x_arr[(self.Pe_x_arr <= -2)]=1.0
            self.alpha_y_arr[(self.Pe_y_arr >= 2)]=0.0
            self.alpha_y_arr[(self.Pe_y_arr <= -2)]=1.0

        elif differencing == "power_law":
            self.alpha_x_arr = alpha_power_law(self.Pe_x_arr)
            self.alpha_y_arr = alpha_power_law(self.Pe_y_arr)

        elif differencing == "central":
            pass

        else:
            raise ValueError("Wrong differencing scheme")

        
        self.__init_fields()
    
    def create_kernel(self, dt = 0.1):
        self.dt = dt
        @ps.kernel
        def kernel_desc():
            # N for North
            c_P = self.c[0,0]
            c_E = self.c[1,0]
            c_W = self.c[-1,0]
            c_N = self.c[0,1]
            c_S = self.c[0,-1]
            
            # concentration gradients
            grad_c_e = c_E - c_P
            grad_c_w = c_P - c_W
            grad_c_n = c_N - c_P
            grad_c_s = c_P - c_S

            # Diffusion flux flux, due to potential gradient
            J_dif_e =   -self.D_x[0,0]*grad_c_e
            J_dif_w =   -self.D_x[-1,0]*grad_c_w
            J_dif_n =   -self.D_y[0,0]*grad_c_n
            J_dif_s =   -self.D_y[0,-1]*grad_c_s

            #alpha depends on Peclet number           
            alpha_e = self.alpha_x[0,0]
            alpha_w = 1.0-self.alpha_x[-1,0]
            alpha_n = self.alpha_y[0,0]
            alpha_s = 1.0-self.alpha_y[0,-1]

            #concentration at faces, with upwind correction    
            c_e = c_E*alpha_e + c_P*(1.0-alpha_e)
            c_w = c_W*alpha_w + c_P*(1.0-alpha_w)
            c_n = c_N*alpha_n + c_P*(1.0-alpha_n)
            c_s = c_S*alpha_s + c_P*(1.0-alpha_s)
            

            # Advection flux, due to potential gradient
            J_adv_e =   -self.D_x[0,0]*self.dU_x[0,0]*c_e
            J_adv_w =   -self.D_x[-1,0]*self.dU_x[-1,0]*c_w
            J_adv_n =   -self.D_y[0,0]*self.dU_y[0,0]*c_n
            J_adv_s =   -self.D_y[0,-1]*self.dU_y[0,-1]*c_s


            J_E = J_dif_e+J_adv_e
            J_W = J_dif_w+J_adv_w
            J_N = J_dif_n+J_adv_n
            J_S = J_dif_s+J_adv_s

            J_tot_current = -J_E + J_W - J_N*self.lambda_n[0,0] +J_S*self.lambda_s[0,0]
            J_tot_prev = -self.div_J[0,0]
            # Adams-Bashforth methods
            J_tot = 3/2*(J_tot_current) - 1/2*(J_tot_prev)
            #J_tot = J_tot_current

            self.grad_c[0,0][0] @= grad_c_e
            self.grad_c[0,0][1] @= grad_c_n

            self.J_dif[0,0][0] @= J_dif_e
            self.J_dif[0,0][1] @= J_dif_n
            self.J_adv[0,0][0] @= J_adv_e
            self.J_adv[0,0][1] @= J_adv_n

            self.J[0,0][0] @= J_E
            self.J[0,0][1] @= J_N

            self.div_J[0,0] @=  -J_tot
            self.c_next[0,0] @= c_P + J_tot*self.dt
        
        self.kernel_desc = kernel_desc
        self.compile_kernel()
    
    def compile_kernel(self):
        gl_spec = [(1, 1),(0, 1)]  # no ghost layer at the bottom boundary
        if __cupy__:
            config = ps.CreateKernelConfig(
                target=ps.Target.GPU, 
                backend=ps.Backend.CUDA,
                ghost_layers=gl_spec,
                default_assignment_simplifications = True
                )
            ast = ps.create_kernel(
                self.kernel_desc,
                config=config,
                )
        else:
            config = ps.CreateKernelConfig(
                cpu_openmp=True,
                ghost_layers=gl_spec,
                default_assignment_simplifications = True
                )
            ast = ps.create_kernel(
                self.kernel_desc,
                config=config,
                )
        self.kernel = ast.compile()
    
    def create_update_loop(self, default_steps, boundary_handler):
        c_tmp_arr = xp.zeros_like(self.c_arr)
        boundary_handler(self)
        def update(steps=None):
            nonlocal c_tmp_arr
            if steps is None: steps = default_steps
            for i in range(steps):
                self.kernel(
                    D_x = self.D_x_arr,
                    D_y = self.D_y_arr,
                    dU_x = self.dU_x_arr,
                    dU_y = self.dU_y_arr,
                    J = self.J_arr,
                    J_dif = self.J_dif_arr,
                    J_adv = self.J_adv_arr,
                    lambda_s = self.lambda_s_arr,
                    lambda_n = self.lambda_n_arr,
                    Pe_x = self.Pe_x_arr,
                    Pe_y = self.Pe_y_arr,
                    c = self.c_arr,
                    c_next = c_tmp_arr,
                    grad_c = self.grad_c_arr,
                    div_J = self.div_J_arr,
                    # div_J_prev = self.div_J_prev_arr,
                    alpha_x = self.alpha_x_arr,
                    alpha_y = self.alpha_y_arr,
                    )
                self.c_arr, c_tmp_arr = c_tmp_arr, self.c_arr
                boundary_handler(self)
            return self.c_arr
        return update
    
    def J_z_tot(self):
        a_z = 2*self.r_arr
        a_z[:,0] = 1/4
        J_z_ = xp.sum(xp.moveaxis(self.J_arr, -1, 0)[0]*a_z, axis = 1)
        J_z_[0] = J_z_[1]
        J_z_[-1] = J_z_[-2]
        return J_z_
    
    def c_z_tot(self):
        a_z = 2*self.r_arr
        a_z[:,0] = 1/4
        c_z_ = xp.sum(self.c_arr*a_z, axis = 1)
        return c_z_
    
    def c_z_average(self):
        a_z = 2*self.r_arr
        a_z[:,0] = 1/4
        c_z_ = self.c_z_tot()/xp.sum(a_z, axis = 1)
        c_z_[0] = c_z_[1]
        c_z_[-1] = c_z_[-2]
        return c_z_
       
    def J_z(self):
        J_z_ = xp.moveaxis(self.J_arr, -1, 0)[0]
        J_z_[0, :] = J_z_[1, :]
        J_z_[-1, :] = J_z_[-2, :]
        J_z_[:, -1] = J_z_[:, -2]
        return J_z_
    
    def grad_c_z(self):
        grad_c_z_ =  xp.moveaxis(self.grad_c_arr, -1, 0)[0]
        grad_c_z_[0, :] = grad_c_z_[1, :]
        grad_c_z_[-1, :] = grad_c_z_[-2, :]
        grad_c_z_[:, -1] = grad_c_z_[:, -2]
        return grad_c_z_
    
    def R_z(self):
        R_z_ = xp.cumsum(
            -xp.gradient(self.c_z_average())/self.J_z_tot()
            )
        return R_z_
    
    def get_current_result(self) -> dict:
        result_fields=[
            "c_arr", "grad_c_arr", "J_arr", "div_J_arr"
            ]
        method_fields=[
            "c_z_average", "J_z", "R_z", "J_z_tot"
            ]
        result_dict = {
            field : getattr(self, field) for field in result_fields
            }
        result_dict.update(
            {
            field : getattr(self, field)() for field in method_fields
            }
        )
        return result_dict
    
    def get_simulation_properties(self) -> dict:
        props = ["dt", "zlayers", "rlayers", "W_arr", "D_arr", "U_arr"]
        result_dict = {
            prop : getattr(self, prop) for prop in props
            }
        return result_dict

    def get_div_J_arr_on_c_arr(self, boundary_handler, dt, smooth_over = 3, steps = 100):
        self.create_kernel(dt)
        loop = self.create_update_loop(
                default_steps=steps,
                boundary_handler=boundary_handler
            )
        def get_J_arr(c_arr_):

            #xp.copyto(self.c_arr, c_arr_)
            self.c_arr = xp.array(c_arr_).reshape((self.zlayers, self.rlayers))
            div_J = xp.zeros_like(self.div_J_arr)
            for _ in tqdm.trange(0, smooth_over):
                loop()
                div_J = div_J + self.div_J_arr
            div_J = div_J/smooth_over
            if __cupy__:
                return float(xp.sum(xp.abs(div_J)).get())
            else:
                return float(xp.sum(xp.abs(div_J)))
        return get_J_arr

    def run_until(
        self, 
        boundary_handler, 
        dt, 
        target_divJ_tot = 1e-6,
        check_every = 10000,
        timeout = 10,
        jump_every = 10,
        max_jump = 1,
        sigmoid_steepness = 1,
        jump_if_change = 1e-3
        ):
        self.create_kernel(dt)
        loop = self.create_update_loop(
                default_steps=check_every,
                boundary_handler=boundary_handler
            )
        start_time = time.time()
        loop()
        if __cupy__:
            div_J_tot = xp.sum(xp.abs(self.div_J_arr)).get()
        else:
            div_J_tot = xp.sum(xp.abs(self.div_J_arr))
        get_elapsed_time = lambda: time.time() - start_time
        is_target_reached = lambda: (get_elapsed_time()>timeout) or (div_J_tot<target_divJ_tot)
        counter = 0
        div_J_tot_old = div_J_tot
        while True:
            loop()
            if __cupy__:
                div_J_tot = xp.sum(xp.abs(self.div_J_arr)).get()
                print(f"sum_divJ = {div_J_tot:.4E}, sum_c = {xp.sum(self.c_z_tot()).get():.4E}")
            else:
                div_J_tot = xp.sum(xp.abs(self.div_J_arr))
                print(f"sum_divJ = {div_J_tot:.4E}, sum_c = {xp.sum(self.c_z_tot()):.4E}")
            if is_target_reached(): break
            counter = counter+1
            if jump_every is not None:
                if counter>=jump_every:
                    if (abs(div_J_tot - div_J_tot_old)/div_J_tot_old)<jump_if_change:
                        print(f"sum_divJ does not change in {jump_every} iterations, jump by {max_jump}")
                        self.c_arr = self.c_arr*sigmoid_k_one_over_k(sigmoid_steepness, max_jump, self.div_J_arr)
                    counter = 0
                    div_J_tot_old = div_J_tot



from pathlib import Path
class SimulationManager:
    def __init__(self, drift_diffusion_kernel, boundary_handler, hdf5_filename) -> None:
        self.DD = drift_diffusion_kernel
        self.update_loop = drift_diffusion_kernel.create_update_loop(
            default_steps = 1000, 
            boundary_handler = boundary_handler
            )
        self.hdf5_filename = Path(hdf5_filename)
        if not self.hdf5_filename.is_file():
            with h5py.File(self.hdf5_filename, "w") as F:
                print("New hdf5 storage created")
                print("File opened")
                self.__create_new_hdf5(F)
            print("File closed")
        else:
            print("Continue previous simulation")
            with h5py.File(self.hdf5_filename, "r") as F:
                print("File opened for read")
                self.__load_last_state(F)
                F.close()
                print("File closed")

    def __load_last_state(self, F):
        for field in [
            "c_arr", "grad_c_arr", "J_arr", "div_J_arr"
            ]:
            print(field)
            xp.copyto(
                getattr(self.DD, field),
                xp.array(F[field][-1])
                )

    def __create_new_hdf5(self, F):
        props = self.DD.get_simulation_properties()
        for prop, value in props.items():
            shape = xp.shape(value)
            if shape:
                try:
                    F.create_dataset(prop, data = value)
                except:
                    F.create_dataset(prop, data = value.get())
            else:
                F.attrs.create(prop, data = value)
        fields = self.DD.get_current_result()
        for field, value in fields.items():
            shape = xp.shape(value)
            dtype = value.dtype
            dset = F.create_dataset(
                field, 
                shape = (1, *shape), 
                maxshape = (None, *shape), 
                dtype = dtype
                )
            try:
                dset[-1]=value
            except TypeError:
                dset[-1]=value.get()

        F.create_dataset("timestep", shape=(1,), maxshape = (None,))
        F["timestep"][-1] = 0

    def __append_current_result(self, F, timestep):
        fields = self.DD.get_current_result()
        for field, value in fields.items():
            dset = F[field]
            dset.resize(dset.shape[0]+1, axis = 0)
            try:
                dset[-1] = value
            except TypeError:
                # TypeError:
                # Implicit conversion to a NumPy array is not allowed. 
                # Please use `.get()` to construct a NumPy array explicitly.
                dset[-1] = value.get()
        
        dset = F["timestep"]
        dset.resize(dset.shape[0]+1, axis = 0)
        dset[-1] = timestep

    def run(self, repeat, steps=10000):
        with h5py.File(self.hdf5_filename, "a") as F:
            print("File is opened to append")
            timestep = F["timestep"][-1]
            for i in tqdm.trange(0,repeat):
                timestep = timestep + steps
                self.update_loop(steps)
                self.__append_current_result(F, timestep)
            F.close()
        print("File closed")


# %%
