import numpy as np 
from math import pi, sin
from numpy.linalg import  solve
from scipy.linalg import inv, eig
from scipy.interpolate import CubicSpline
from scipy.special import erf 
from foruier import toeplitz_0;
from layers_utils import phys_val, pairwise, float_parts, average_max1, average8cm
from copy import deepcopy
from matplotlib import pyplot as plt
import logging
from numba import njit 
import time 
try:                              # SciPy >= 1.14 renamed these (M14)
    from scipy.integrate import trapezoid as trapz, simpson as simps
except ImportError:                   # SciPy < 1.6
    from scipy.integrate import trapz, simps
from scipy.integrate import quad
import progressbar
from foruier import get_eps_smoothed_coefs
import pdb

@njit
def eps_i(m,f):
    delta = 0
    p = 1.0-delta
    if m==0:
        return f
    else:
        return np.sin(pi*m*f)/(pi*m)*(p**np.abs(m))



logging.basicConfig(filename='RCWA.log', level=logging.ERROR)
logger = logging.getLogger()
#spead of light, cm/s
c_light = 2.99792458e10  # cm/s, exact SI value
v = 1e12
m0 = 9.11e-28 # g
m0kg = 9.11e-31 # kg
elCI= 1.602e-19  
el = 4.8e-10 # CGSE
meff = 0.26
# k0 = 2*pi*v/c_light
#np.set_printoptions(precision=1) 

class InSb:
    def __init__(self,
                eps0 = 16, #17.9,
                A = 2.6,
                Eg = 0.18): #2.6
        self.eps0 = eps0
        self.A = A
        self.Eg = Eg

    def epsInSb(self,v,units='Hz'):
        A = self.A
        hp_eV = 4.14E-15 #eV*s
        Eg = 0.18 #0.228
        x=hp_eV*v/Eg
        ReEps, ImEps = self.eps0,0.
        if (x<1):
            ReEps+=A*(2-np.sqrt(1+x)-np.sqrt(1-x))/x**2
        else:
            ImEps+=A*np.sqrt(x-1)/x**2 
            ReEps+=A*(2-np.sqrt(1+x))/x**2
        return ReEps+1j*ImEps
InSb_default = InSb()

epsInSb = InSb_default.epsInSb
tau_s = lambda mu,meff: meff*m0kg*mu/elCI
def sigma(N3d,mu,meff):
    return el*el*N3d*tau_s(mu,meff)/(meff*m0)

def savearr(arr,fname):
    arr = np.column_stack(arr)
    np.savetxt(fname,arr)
 

# @njit
def qm_fill(period,N,k0):
    q0 = 2*pi/period/k0
    return q0*np.arange(-N,N+1,dtype=np.complex128)     

@njit   # intentional: this decorator used to float above a commented-out
        # version of the function, which made it look accidental (M9)
def km_fill(qm, eps_j):
    k=(qm*qm).astype(np.complex128)
    k_m = np.sqrt(k-eps_j)
    k_m1 = np.sqrt(eps_j-k)
    k_m = np.where(np.real(k-eps_j)>0,k_m,-1j*k_m1)
    return np.diag(k_m)

@njit
def km_fill_1(qm, eps_j):
    #print('qm=',qm)
    k=(qm*qm).astype(np.complex128)
    k_m = np.sqrt(k-eps_j)
    k_m1 = np.sqrt(eps_j-k)
    k_m = np.where(np.real(k-eps_j)>0,k_m,-1j*k_m1)
    return k_m

def eps_xz(x,z, eps_i, zi, plates, fill, period):#
    """Signature:
        x,z - spatial coordinates,
        eps_i - array (N_layer,2),
        zi - interfaces  array,
        plates - array if layer is plate
        fill - filling factor"""
    # try:
    eps = np.ones(x.shape, dtype = np.complex128)
    eps_z = np.ones(x.shape[0],dtype = np.complex128)
    eps_z_op = np.ones(x.shape[0],dtype = np.complex128)
    _z = z[:,1]
    #print(_z.shape)
    # import pdb
    # pdb.set_trace()
    for i in range(len(zi)-1):
        z0=zi[i]
        z1=zi[i+1]
        # final interface inclusive, otherwise a sample landing exactly on the
        # bottom of the structure keeps eps = 1 (finding H1)
        mask = (_z>=z0) & (_z<=z1) if i==len(zi)-2 else (_z>=z0) & (_z<z1)
        eps_z[mask] = eps_i[i,0]
        eps_z_op[mask] = eps_i[i,0]
        if not plates[i]:
            eps_z_op[mask] = eps_i[i,1]
    m, n = x.shape
    # print(f'period = {period}')
    # print(fill*period)
    x_idx = np.arange(n)[x[0]<=fill*period][-1]+1
    # print(x_idx)
    # print(x[0].max())
    # print(eps_z)
    # print(eps_z_op)
    eps[:,:x_idx] = np.broadcast_to(eps_z[:,None],(m, x_idx) )
    eps[:,x_idx:] = np.broadcast_to(eps_z_op[:,None],(m, n-x_idx) )
    # except Exception as e:
    #     print(e)
    #     pdb.set_trace()
    return eps

def eps_xz_new(x, z, eps_i, zi, plates, fill_arr, period):
    """
    Compute spatial distribution of dielectric constant for a grating structure with multiple media and openings.

    Args:
        x, z: Spatial coordinates.
        eps_i: Array of dielectric constants for each layer (N_layer, N_media + 1). The last value is for the opening.
        zi: Array of z-coordinates for interfaces.
        plates: Array indicating if layer is a plate.
        fill_arr: Array of filling factors for each medium in the period. Does not include the opening.
        period: Period of the grating.

    Returns:
        2D array of dielectric constant values.
    """
    eps = np.ones(x.shape, dtype=np.complex128)  # Initialize the array with complex ones
    _z = z[:, 1]  # Extracting the z-axis values

    for i in range(len(zi) - 1):
        z0, z1 = zi[i], zi[i + 1]  # Start and end points of the current layer
        # final interface inclusive (finding H1)
        mask = (_z >= z0) & (_z <= z1) if i == len(zi)-2 else (_z >= z0) & (_z < z1)
        eps_layer = np.zeros(x.shape[1], dtype=np.complex128)  # Initialize layer's eps array

        if not plates[i]:  # Check if the layer is not a plate
            x_pos = 0
            for fill, eps_val in zip(fill_arr[i], eps_i[i]):
                x_max = fill * period  # Calculate the maximum x position for the current fill
                x_mask = (x[0] >= x_pos) & (x[0] < x_max)  # Mask for the current fill in x-axis
                eps_layer[x_mask] = eps_val  # Assign the dielectric constant for the current fill
                x_pos = x_max  # Update the x position for the next fill

            # Handle the opening
            opening_mask = x[0] >= x_pos  # Mask for the opening
            eps_layer[opening_mask] = eps_i[i][-1]  # Assign the dielectric constant for the opening
        else:
            eps_layer[:] = eps_i[i][0]  # If it's a plate, assign the same dielectric constant across the layer
        N = np.sum(mask)
        # print(N)
        # pdb.set_trace()
        eps[mask] = np.broadcast_to(eps_layer[None,...], (N, eps_layer.shape[0]))  # Assign the calculated eps values to the main array
        # eps[:, mask] = eps_layer
        
    return eps


def fourier_frame_x(x_norm, x_shift):
    """Map a normalised plotting abscissa to the Fourier frame [-0.5, 0.5).

    The field routines evaluate the harmonics at X - x_shift*period, so the
    permittivity map and the loss masks must use the same origin, otherwise
    every layer whose fill differs from Layers[0] is drawn in the wrong place.
    """
    return (x_norm - x_shift + 0.5) % 1.0 - 0.5


def eps_xz_segments(x, z, segments, zi, period, x_shift=0.0):
    """Real-space permittivity map driven by per-layer segment descriptions.

    Args:
        x, z:     meshgrids; x in the same length unit as `period`, z as `zi`.
        segments: segments[i] is Layer.eps_segments() for layer i.
        zi:       layer interfaces.
        period:   grating period, same unit as x.
        x_shift:  normalised offset of the plotting frame w.r.t. the Fourier
                  frame (see fourier_frame_x).

    Every layer is drawn with its OWN profile, and a strip that wraps across
    the period edge is handled because the abscissa is wrapped, not clipped.
    """
    eps = np.ones(x.shape, dtype=np.complex128)
    _z = z[:, 0]
    xn = fourier_frame_x(x[0]/period, x_shift)
    n_layers = len(zi) - 1
    for i in range(n_layers):
        z0, z1 = zi[i], zi[i+1]
        # The final interface is inclusive: a sample landing exactly on the
        # bottom of the structure must still belong to the last layer, or it
        # keeps eps = 1 and |Ex|^2 = |Dx/1|^2 blows up there.
        if i == n_layers - 1:
            zmask = (_z >= z0) & (_z <= z1)
        else:
            zmask = (_z >= z0) & (_z < z1)
        if not zmask.any():
            continue
        # start from the outermost medium so that round-off at a segment edge
        # can never leave a sample unassigned
        row = np.full(x.shape[1], segments[i][0][2], dtype=np.complex128)
        for x0, x1, eps_s in segments[i]:
            if x1 <= x0:
                continue
            row[(xn >= x0) & (xn < x1)] = eps_s
        eps[zmask] = row
    return eps


def sqrt_(x):
    x=np.sqrt(x)
    return x #np.where(np.real(x)>0,x,-x)

class _NullBar:
    """Inert stand-in for progressbar.ProgressBar when verbose is False.

    Exists so that every spectrum driver can have exactly ONE loop body.  The
    previous `if verbose: <loop> else: <loop>` pattern kept two copies of each
    driver, and the copies drifted: the quiet branches variously dropped the
    incidence angle, passed the wrong argument as Neq, or called a progress bar
    that was never created (findings H2-H5).
    """
    def update(self, *args, **kwargs):
        pass

    def start(self):
        return self

    def finish(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def progress_bar(max_value, verbose):
    """A real progress bar when verbose, an inert one otherwise."""
    if verbose:
        return progressbar.ProgressBar(max_value=max_value)
    return _NullBar()


def get_layer_id(z, z_int):
    for v, k in enumerate(pairwise(z_int)):
        if z>=k[0] and z<=k[1]:
            return v

class GratingLayer:

    """Implement Grating instance.
    Default  parameters 
    period=1, fill=0.5, depth=10,
    eps0=1.0, eps1=1.0, sigma0=0, tau = 0
    """
    def __init__(self, period=1, fill=0.5, depth=10,
                 eps0=1.0, eps1=None, sigma0=0, sigma0_op=0, tau = 0, tau_op=0,
                 plate=False, 
                 meff=0.26,
                 dispersion_type=None,
                 material_type = None,
                 material = {},
                 optimized = True,
                 smoothed = False,
                 delta = 0.01, # smoothing factor
                 name = None):
        # super().__init__(eps=eps1, sigma0=sigma0, tau=tau, meff=meff)
        self.layer_type = 'GratingLayer'
        self.smoothed = smoothed
        self.delta = delta
        self.period = period*1e-4
        self.fill = fill 
        self.depth = depth*1e-7
        self.eps0 = eps0
        self.eps1 = eps1 if not eps1 is None else eps0
        if material_type=='InSb':
            if len(material.keys())>0:
                self.eps1 = InSb(**material).epsInSb
                print('InSb layer init with params:')
                for k, v in material.items():
                    print(f'{k}={v}')
            else:
                self.eps1 = InSb().epsInSb
                print('InSb layer init with DEFAULT params:')
                for k, v in InSb().__dict__.items():
                    print(f'{k}={v}')

            
        self.sigma0 = sigma0
        self.sigma0_op = sigma0_op
        self.tau = tau
        self.tau_op = tau_op
        self.trans_matrix = None
        self.plate = plate
        self.meff = meff
        self.dispersion_type = dispersion_type
        self.material = None
        self.D0 = None
        self.W = None 
        self.V = None
        self.L = None 
        self.l = None
        self.C_plus = None
        self.C_minus = None
        self.eps_mm = None
        self.optimized = optimized
        self.name = name
        self.debug = dict(k=[],
                          err=[],
                          q=[],
                          eps1=[])
        
        # self.phi = k0*depth*1e-7

    def sigma(self,v):
        w=2*pi*v
        return self.sigma0/(1-1j*w*self.tau)

    def sigma_op(self,v):
        w=2*pi*v
        return self.sigma0_op/(1-1j*w*self.tau_op)

    @classmethod
    def PlateLayer(cls, eps,  depth=10, depth_mkm=None,
                  sigma0=0, tau = 0, 
                  meff=0.26,
                  dispersion_type=None,name=None, period=1 ):
                 depth_nm = depth
                 if depth_mkm is not None:
                     depth_nm = depth_mkm*1000
                 return cls(eps1=eps, 
                            plate = True,
                            sigma0=sigma0, 
                            tau=tau,  
                            meff=meff,  
                            depth=depth_nm,
                            period=period,
                            dispersion_type= dispersion_type,
                            name = name)

    @classmethod
    def from_mobility(cls, mu, n, mu_op=0, n_op=0,
                        eps1=None, 
                        eps0=1, 
                        meff=0.26,  
                        period=1,   
                        fill=0.5, 
                        depth_mkm=1,
                        plate=False,
                        material_type = None,
                        material = {},
                        dispersion_type=None,
                        name = ''):
        """mu - cm^2/(V*s), n- cm^{-3}, depth - mkm """
        mu*=1e-4 #conversion to m^2
        mu_op*=1e-4
        tau = meff*m0kg*mu/elCI #tau_s(mu,meff)
        tau_op = meff*m0kg*mu_op/elCI #tau_s(mu,meff)
        # print(f'tau = {tau*1e12}')
        sigma0 = sigma(n,mu,meff)
        sigma0_op = sigma(n_op,mu_op,meff)
        # print(f'sigma0 = {sigma0}')
        # print(f'sigma0_op = {sigma0_op}')
        return cls(eps1=eps1, 
                    eps0=eps0, 
                    sigma0=sigma0, 
                    sigma0_op=sigma0_op, 
                    tau=tau,  
                    tau_op=tau_op,  
                    meff=meff,  
                    period=period, 
                    fill=fill, 
                    depth=depth_mkm*1e3,
                    plate=plate, 
                    dispersion_type= dispersion_type,
                    material_type = material_type,
                    material = material,
                    name = name)
    @property
    def f_plasma(self):
        return np.sqrt(4*pi*self.N3d*el**2/(self.eps_1*self.meff*m0))/(2*pi)
    
    def f_plasma2D_mikh(self, eps):
        x = (np.pi*self.fill)**2
        w = self.period*self.fill
        f0 = (2/np.pi)*np.sqrt(self.sigma0*self.depth/(eps*w*self.tau))   
        print(f0)
        f_plasma = f0*np.sqrt(1-x/24-x**2/960)
        return f_plasma

    @property
    def depth_mkm(self):
        return self.depth*1e4

    @depth_mkm.setter
    def depth_mkm(self,d):
        self.depth=d*1e-4

    @property
    def period_mkm(self):
        return self.period*1e4

    @period_mkm.setter
    def period_mkm(self,d):
        self.period=d*1e-4
    

    @property
    def mu(self):
        # self.meff, not the module-level default (M6)
        return 1e4*elCI*self.tau/(m0kg*self.meff)

    @mu.setter
    def mu(self,muCI):
        assert self.mu!=0, 'mobility 0 !'
        k = muCI/self.mu
        self.tau *= k
        self.sigma0 *= k        
    
    @property
    def N3d(self):
        try:
            return self.meff*m0 * self.sigma0 / (self.tau * el**2)
        except Exception as e:
            print(e)
            return 0

    @N3d.setter
    def N3d(self,n):
        assert self.N3d!='sigma must be >0'
        self.sigma0 *= n/self.N3d 

    def eps_v(self,v=None):
        if v is not None:
            if callable(self.eps1): #v is not None and 
                return self.eps1(v)+2j*self.sigma(v)/v
            else:
                return self.eps1+2j*self.sigma(v)/v
        # if self.dispersion_type is None:
        #     if v is None:
        #         return self.eps1
  
        # elif self.dispersion_type=='InSb':
            
        #     return epsInSb(v)
        # else:
        
        return self.eps1
 
    def eps_v_op(self,v=None):
        """epsilon(v) in opening beatween stips"""
        if v is not None and callable(self.eps0):
              return self.eps0(v)+2j*self.sigma_op(v)/v
        if self.dispersion_type is None:
            # gate on the OPENING's conductivity, not the strip's tau (M1)
            if v is None or self.sigma0_op==0:
                return self.eps0
            return self.eps0+2j*self.sigma_op(v)/v
        elif self.dispersion_type=='InSb':
            return epsInSb(v)
        else:
            return self.eps0



    def eps_1(self,N, v=None):
        eps1 = self.eps_v(v=v)
        if self.plate:
            # A plate is uniform: return the full coefficient set rather than a
            # length-1 array, which keeps optimized=False usable (M8).  self.eps0
            # is no longer overwritten here -- that replaced a callable eps0 with
            # its value at whichever frequency ran last (M5).
            f_coefs = np.zeros(2*N+1, dtype=np.complex128)
            f_coefs[N] = eps1
            return f_coefs
        else:
            eps0 = self.eps_v_op(v=v)
        if self.smoothed:
            f_coefs = get_eps_smoothed_coefs(N, [self.fill],[eps1, eps0], dx = self.delta )
        else:
            eps = lambda x: eps_i(x,self.fill)
            eps = np.vectorize(eps)
            f_coefs = (eps1-eps0)*eps(np.arange(-N,N+1))
            f_coefs[N] += eps0
        return f_coefs  
 
    def eps_inv(self,N, v=None):
        eps = lambda x: eps_i(x, self.fill)
        eps = np.vectorize(eps)
        eps1 = self.eps_v(v)
        if self.plate:
            f_coefs = np.zeros(2*N+1, dtype=np.complex128)
            f_coefs[N] = 1.0/eps1
            return f_coefs
        eps0 = self.eps_v_op(v=v)
        assert eps0 != 0, 'eps0 must be positive'
        assert eps1 != 0, 'eps1 must be positive'
        eps0=1/eps0
        eps1=1/eps1
        if self.smoothed:
            f_coefs = get_eps_smoothed_coefs(N, [self.fill],[eps1, eps0], dx = self.delta )
        else:
            f_coefs = (eps1-eps0)*eps(np.arange(-N,N+1))
            f_coefs[N] += eps0
        return f_coefs  
 
 
    def eps_segments(self, v=None):
        """Profile along one period, as an ordered partition of x' in [-0.5,0.5).

        Returns [(x0, x1, eps), ...] with x0 < x1, covering the period exactly.
        The convention is the one eps_1()/eps_inv() use: strips are CENTRED on
        x' = 0.  Having a single description means the real-space map and the
        loss integral cannot disagree with the Fourier coefficients the solver
        was given.
        """
        if self.plate:
            return [(-0.5, 0.5, self.eps_v(v))]
        f = 0.5*self.fill
        eps1, eps0 = self.eps_v(v), self.eps_v_op(v)
        return [(-0.5, -f, eps0), (-f, f, eps1), (f, 0.5, eps0)]

    def qm_fill(self,N,k0):
        q0 = 2*pi/self.period/k0
        return q0*np.arange(-N,N+1)

    def WV_TE(self,q_m,eps_m,N):
        A =  np.eye(2*N+1)*q_m**2- toeplitz_0(eps_m)
        L, W = eig(A)
        L = sqrt_(L.astype(np.complex128))
        V =  W * L
        return W,V,L

    def WV_TM(self,q_m,eps_m,N,v):
        eps_inv_m = self.eps_inv(2*N,  v=v)
        eps_mm = inv(toeplitz_0(eps_m))
        t_eps_inv_m = toeplitz_0(eps_inv_m)
        eps_inv_mm = inv(t_eps_inv_m)
        B = q_m[:,None]*eps_mm*q_m-np.eye(2*N+1)
        B = eps_inv_mm.dot(B)
        L, W = eig(B)
        # assert np.all(np.isclose((W @ np.diag(L) @ inv(W)),B)) , 'Eigenvector decomposition failed!!'
        # err = np.abs(W @ np.diag(L) @ inv(W) - B).sum()
        # self.debug['err'].append(err)
        L = sqrt_(L.astype(np.complex128))
        V = np.dot(t_eps_inv_m, W*L)
        self.eps_mm = eps_mm
        self.B = B
        return W,V,L
  
    def layer_fg(self,period,k0,N,fg_in, k0_inc ,polarization='TM', dispersion=True):
        v = k0*c_light/(2*pi) if dispersion  else None
        if N==0:
            q_m = np.array([0.0], dtype=np.complex128) + k0_inc
        else:
            q_m = qm_fill(period, N, k0) + k0_inc
        eps_m = self.eps_1(2*N, v=v)
        eps1 = self.eps_v(v)
        self._eps1 = eps1
        # self.debug['eps1'].append(eps1)
        # self.debug['q'].append(q_m)
        # pdb.set_trace()
        if self.optimized and self.plate:
            W = np.eye(2*N+1)
            L = km_fill_1(q_m, eps1)
            # Admittance of a homogeneous layer is polarization dependent:
            # TM (H-field formulation) carries the 1/eps factor, TE does not.
            # The TM expressions are left exactly as they were so that TM
            # results stay bit-identical.
            if polarization=='TM':
                V = 1j*np.diag(L/eps1)
                Vinv = -1j*np.diag(eps1/L)
            else:
                V = 1j*np.diag(L)
                Vinv = -1j*np.diag(1.0/L)
            if N==0:
                A1 = 0.5*np.array([[1.0,Vinv[0,0]],
                                   [1.0,-Vinv[0,0]]]) 
            else:
                A1 = 0.5*np.block([[W,Vinv],
                                  [W,-Vinv]])
            self.eps_mm = np.eye(2*N+1)/eps1
        else:
            if polarization=='TM':
                W,V,L =  self.WV_TM(q_m,eps_m,N,v)
            else:
                W,V,L =  self.WV_TE(q_m,eps_m,N)
            V*=1j   
            # A1 = np.block([[W,W],
            #             [V,-V]])
            W_1 = inv(2*W)
            V_1 = inv(2*V)
            A1 = np.block([[W_1,V_1],
                        [W_1,-V_1]])
            # A1 = inv(A1)
        # self.debug['k'].append(L.copy())
        L=L.astype(np.complex128) #np.clongdouble
        # print('W = ',W)
        # print('V = ',V)
        # print('L = ', L)
        #print(f'{self.name} Re L:', np.real(L))
        #print(f'{self.name} Im L:', np.imag(L),'\n')
        X = np.exp(-L*k0*self.depth)
        I = np.eye(2*N+1,dtype = np.complex128)
        # fg = fg_in #np.vstack((f_in,g_in))

        ab = A1.dot(fg_in)
        a = ab[0:2*N+1]
        if N==0:
            a1= 1.0/a
        else: 
            a1= inv(a)
        b = ab[2*N+1:4*N+3]
        #Xd = np.diag(X)
        #a_1Xd = np.dot(a1,Xd)
        a_1Xd = a1*X
        D0 = b.dot(a_1Xd)
        self.D0 = D0
        self.W = W
        self.V = V
        self.L = L
        #Xd = np.diag(X)
        #D = Xd.dot(D0)
        D = X[:,None]*D0
        if self.optimized and self.plate:
            fg = np.vstack((I+D,
                            V.dot(I-D)))
        else:
            fg = np.vstack((W.dot(I+D),
                            V.dot(I-D)))
        return fg,a_1Xd

    def calc_field_coefs(self, Tj):
        N = self.D0.shape[0]
        ID0 =np.vstack((np.eye(N, dtype=self.D0.dtype),
                      self.D0)) 
        C = ID0.dot(Tj)
        self.C_plus = C[:N]
        self.C_minus = C[N:]

    def Exm_z(self, z, l): 
        Lam_plus = np.diag(np.exp(-l*z))@self.C_plus
        Lam_minus = np.diag(np.exp(l*(z-self.depth)))@self.C_minus
        LC_Ex = Lam_plus - Lam_minus
        LC_H = Lam_plus + Lam_minus
        Em_z = self.V@LC_Ex
        LC_H = self.W@LC_H
        return np.squeeze(Em_z), np.squeeze(LC_H)
 
class GratingLayer2(GratingLayer):
    def __init__(self, period=1, fill=0.5, fill2 = 0.6, depth=10, eps0=1, eps1=None, eps2= None,
                sigma0=0, sigma0_op=0, tau=0, tau_op=0, plate=False, meff=0.26,
                dispersion_type=None, material_type=None, material={},
                optimized=True, smoothed=False, delta=0.01, name=None):
        # Keyword arguments, not positional: the parent gained `smoothed` and
        # `delta` before `name`, so the old positional call passed `name` into
        # `smoothed` -- a named layer silently switched eps_1 to the smoothed
        # branch and lost its name (finding H7).
        super().__init__(period=period, fill=fill, depth=depth, eps0=eps0, eps1=eps1,
                         sigma0=sigma0, sigma0_op=sigma0_op, tau=tau, tau_op=tau_op,
                         plate=plate, meff=meff, dispersion_type=dispersion_type,
                         material_type=material_type, material=material,
                         optimized=optimized, smoothed=smoothed, delta=delta,
                         name=name)
        self.layer_type = 'GratingLayer2'
        self.eps2 = eps0 if eps2 is None else eps2
        self.fill2 = fill2

    def eps_1(self,N, v=None):
        # import pdb; pdb.set_trace()
        eps1 = self.eps_v(v=v)
        if callable(self.eps2):
            eps2 = self.eps2(v)
        else:
            eps2 = self.eps2

        if self.plate:
            # A plate is uniform: return the full coefficient set rather than a
            # length-1 array, which keeps optimized=False usable (M8).  self.eps0
            # is no longer overwritten here -- that replaced a callable eps0 with
            # its value at whichever frequency ran last (M5).
            f_coefs = np.zeros(2*N+1, dtype=np.complex128)
            f_coefs[N] = eps1
            return f_coefs
        else:
            eps0 = self.eps_v_op(v=v)
        if self.smoothed:
            f_coefs = get_eps_smoothed_coefs(N, [self.fill, self.fill2],[eps2, eps1, eps0], dx = self.delta )
        else:
            _eps1 = lambda x: eps_i(x,self.fill)
            _eps1 = np.vectorize(_eps1)
            _eps2 = lambda x: eps_i(x,self.fill2)
            _eps2 = np.vectorize(_eps2)
            f_coefs = (eps1-eps0)*_eps2(np.arange(-N,N+1))
            f_coefs += (eps2-eps1)*_eps1(np.arange(-N,N+1))
            f_coefs[N] += eps0
        return f_coefs  
        
    def eps_segments(self, v=None):
        """Nested three-medium profile: core eps2 of width fill inside a shell
        eps1 of width fill2, on a background eps0.  Matches eps_1()."""
        if self.plate:
            return [(-0.5, 0.5, self.eps_v(v))]
        f1, f2 = 0.5*self.fill, 0.5*self.fill2
        eps2 = self.eps2(v) if callable(self.eps2) else self.eps2
        eps1, eps0 = self.eps_v(v), self.eps_v_op(v)
        return [(-0.5, -f2, eps0), (-f2, -f1, eps1), (-f1, f1, eps2),
                (f1, f2, eps1), (f2, 0.5, eps0)]

    def eps_inv(self,N, v=None):
        _eps1 = np.vectorize(lambda x: eps_i(x,self.fill))
        _eps2 = np.vectorize(lambda x: eps_i(x,self.fill2))
        eps1 = self.eps_v(v)
        if self.plate:
            f_coefs = np.zeros(2*N+1, dtype=np.complex128)
            f_coefs[N] = 1.0/eps1
            return f_coefs
        eps0 = self.eps_v_op(v=v)
        assert eps0 != 0, 'eps0 must be positive'
        assert eps1 != 0, 'eps1 must be positive'
        eps0=1/eps0
        eps1=1/eps1
        if callable(self.eps2):
            eps2 = 1/self.eps2(v)
        else:
            eps2=1/self.eps2
        if self.smoothed:
            f_coefs = get_eps_smoothed_coefs(N, [self.fill, self.fill2],[eps2,eps1, eps0], dx = self.delta )
        else:
            f_coefs = (eps1-eps0)*_eps2(np.arange(-N,N+1))
            f_coefs += (eps2-eps1)*_eps1(np.arange(-N,N+1))
            f_coefs[N] += eps0
        return f_coefs       

class GratingLayerN(GratingLayer):
    def __init__(self, period=1, fill_arr = [0.5], eps_arr = [1,1],
                depth=10,
                eps0=1,
                sigma_arr= None,
                tau_arr= None,
                plate=False, meff=0.26, 
                dispersion_type=None, 
                material_type=None, 
                material={}, optimized=True, name=None):
        if len(eps_arr)-1 != len(fill_arr):
            raise ValueError(f"len of eps_arr must be len of fill_arr + 1")
        self.layer_type = 'GratingLayerN'
        self.period = period*1e-4
        self.depth = depth*1e-7
        self.eps_arr = eps_arr
        self.fill_arr = fill_arr
        self.fill = fill_arr[-1]
        self.sigma_arr = sigma_arr if not sigma_arr is None else np.zeros((len(eps_arr),))
        self.tau_arr = tau_arr if not tau_arr is None else np.zeros((len(eps_arr),))
        self.optimized = optimized
        self.name = name
        self.meff = meff
        self.trans_matrix = None
        self.plate = plate
        self.dispersion_type = dispersion_type
        self.material = None
        self.D0 = None
        self.W = None 
        self.V = None
        self.L = None 
        self.C_plus = None
        self.C_minus = None
        self.eps_mm = None
        self.optimized = optimized
        self.name = name
        self.eps1 = eps_arr[0]
        self.debug = dict(k=[],
                          err=[],
                          q=[],
                          eps1=[])
        
    def eps_v(self,v=None):
        if v is not None:
            w=2*pi*v
            eps_v_arr = np.zeros((len(self.eps_arr)), dtype=np.complex128)
            for idx, eps_i in enumerate(self.eps_arr):
                eps_v_arr[idx] = eps_i(v)  if callable(eps_i) else eps_i
                sigma_v = self.sigma_arr[idx]/(1-1j*w*self.tau_arr[idx])
                eps_v_arr[idx] += 2j*sigma_v/v
        else:
            eps_v_arr = self.eps_arr
        return eps_v_arr
    
    def eps_1(self,N, v=None):

        f_coefs = np.zeros((2*N+1,), dtype=np.complex128)
        eps_arr_v = self.eps_v(v)
        for idx, fill in enumerate(self.fill_arr):
            eps = lambda x: eps_i(x,fill)
            eps = np.vectorize(eps)
            f_coefs+=(eps_arr_v[idx]-eps_arr_v[idx+1])*eps(np.arange(-N,N+1))
        f_coefs[N] += eps_arr_v[-1]
        return f_coefs
    
    def eps_segments(self, v=None):
        """Nested profile built from the cumulative widths in fill_arr.

        This mirrors eps_1(), which sums centred strips and therefore describes
        nested slabs, NOT the sequential regions eps_xz_new() used to draw.
        Matching eps_1() keeps the map consistent with the solver; whether
        fill_arr *ought* to mean nested or sequential is a separate question
        that would change eps_1() and this method together.
        """
        if self.plate:
            return [(-0.5, 0.5, np.atleast_1d(self.eps_v(v))[0])]
        eps = self.eps_v(v)
        h = [0.5*f for f in self.fill_arr]
        segs = [(-0.5, -h[-1], eps[-1])]
        for i in range(len(h)-1, 0, -1):
            segs.append((-h[i], -h[i-1], eps[i]))
        segs.append((-h[0], h[0], eps[0]))
        for i in range(len(h)-1):
            segs.append((h[i], h[i+1], eps[i+1]))
        segs.append((h[-1], 0.5, eps[-1]))
        return segs

    def eps_inv(self,N, v=None):
        # import pdb
        # pdb.set_trace()
        f_coefs = np.zeros((2*N+1,), dtype=np.complex128)
        eps_arr_v = self.eps_v(v)
        eps_arr_v = 1/eps_arr_v
        for idx, fill in enumerate(self.fill_arr):
            eps = lambda x: eps_i(x,fill)
            eps = np.vectorize(eps)
            f_coefs+=(eps_arr_v[idx]-eps_arr_v[idx+1])*eps(np.arange(-N,N+1))
        f_coefs[N] += eps_arr_v[-1]
        return f_coefs
     


class GratingStructure:
    """docstring for GratingStructure"""
    def __init__(self, Layers, eps_inp = None, eps_out = 1, name = ''):
        self.Layers = Layers
        self.eps_inp = Layers[0].eps0 if eps_inp is None else eps_inp
        self.eps_out = eps_out
        self.titles = dict(
            mu=phys_val(r'$\mu$',r' $cm^2/V\times s$'),
            N3d=phys_val(r'$n_{3D}$',r' $cm^{-3}$'),
            depth_mkm=phys_val(r'$D$',r' $\mu m$')
            )
        try:
            self.fill = self.Layers[0].fill
        except:
            print('No fill value')
        # Only layers with lateral structure need a common period; a plate is
        # uniform in x, so its `period` is meaningless and PlateLayer leaves it
        # at the default (M7).
        periods = {round(float(L.period), 15) for L in Layers if not L.plate}
        if len(periods) > 1:
            raise ValueError('all patterned layers must share one period; got '
                             + ', '.join(f'{q*1e4:g} um' for q in sorted(periods)))
        z_int = np.zeros(len(Layers)+1)
        for idx,Layer in enumerate(Layers):
            z_int[idx+1]+= z_int[idx]+Layer.depth
        self.z_int = z_int
        self.spectrA = None
        self.spectrT = None
        self.spectrR = None
        self.name = name


    def __repr__(self):
            Ndig,Nexp = float_parts(self.Layers[0].N3d)
            Ndig=Ndig.rstrip('0')
            it = dict(
            mu = f"{self.Layers[0].mu:4.0f}",
            N = '_'.join([Ndig,Nexp]),
            d = f"{self.Layers[0].depth_mkm}",
            a = f"{self.Layers[0].period*1e4:8.0f}".lstrip(' '),
            f = f"{self.Layers[0].fill:.2f}".strip('0.'),
            D = f"{self.Layers[0].depth_mkm:4.0f}".lstrip(' '),
            )
            rep = '!'.join([key+value for key,value in it.items()])
            return rep    
    
    @property
    def x_shift(self):
        """Normalised offset between the plotting frame and the Fourier frame.

        The field routines evaluate the harmonics at X - Layers[0].fill*period/2;
        the permittivity map and the loss masks must use the same origin.
        """
        return 0.5*getattr(self.Layers[0], 'fill', 0.0)

    def eps_structure(self, X, Y, v):
        """Spatial distribution of the permittivity on the (X, Y) meshgrid.

        X is in micrometres and Y in centimetres, matching the field routines.
        Each layer is drawn from its own eps_segments(), so mixed stacks and
        per-layer fill factors are handled without any layer_type dispatch.
        """
        segments = [Layer.eps_segments(v) for Layer in self.Layers]
        return eps_xz_segments(X, Y, segments, self.z_int,
                               self.Layers[0].period*1e4, x_shift=self.x_shift)

    def solver(self,k0,N, Theta = 0, dispersion=True,polarization='TM',
              field = False):
        k0_inc =  np.sqrt(self.eps_inp)*sin(Theta*pi/180)
        period = self.Layers[0].period
        qm = k0_inc + qm_fill(period, N, k0) 
        I = np.eye(2*N+1,dtype = np.complex128)
        Ic0 = km_fill(qm, self.eps_inp)
        IcN = km_fill(qm, self.eps_out)
        if polarization=='TM':
            fg = np.vstack((I,1j*IcN/self.eps_out))
            # The reflected row needs the same 1/eps factor its transmitted
            # counterpart has: in TM the tangential field is
            # E_x = -k_zm r_m / eps_inp.  Without it the match at z = 0 is
            # inconsistent and energy is not conserved for eps_inp != 1.
            fg1 = np.vstack((-I,1j*Ic0/self.eps_inp))
        else:
            fg = np.vstack((I,1j*IcN)) # 1j*IcN
            fg1 = np.vstack((-I,1j*Ic0))
        for Layer in reversed(self.Layers):
            fg, trans_l = Layer.layer_fg(period, k0,N,fg, k0_inc, dispersion=dispersion, polarization=polarization)
            Layer.trans_matrix = trans_l
        
        M = np.block([fg,fg1])

        if polarization=='TM':
            C = np.hstack([I[:,N],
                        I[:,N]*np.cos(Theta*pi/180)/np.sqrt(self.eps_inp)])
        else:
            C = np.hstack([I[:,N],
                        I[:,N]*np.cos(Theta*pi/180)*np.sqrt(self.eps_inp)])
        # print('M=',M)
        # print('C=',C)
        C = C[:,None]
        tr = solve(M,C)
        t = tr[0:2*N+1]
        r = tr[2*N+1:4*N+3] 
        for Layer in self.Layers:
            if field:
                Layer.calc_field_coefs(t)
            t =Layer.trans_matrix.dot(t)
        self.r = r
        self.t = t
        #print('r=',r)
        # km = np.diag(Ic)
        # self.Em_z0 = -1j*km[:,None]*self.r
        # self.Em_z0[N] += 1
        return t,r



    def get_field_coefs_on_z(self, z, k0, km, l_free, N, z_int, theta=0):
         
        if z<0:
            Hm_z = np.exp(l_free*z)[:,None]*self.r
            Hm_z[N] += np.exp(-l_free[N]*z)
            Em_z = -1j*(km*np.exp(l_free*z))[:,None]*self.r/self.eps_inp
            Em_z[N] += np.cos(theta)/np.sqrt(self.eps_inp)*np.exp(-l_free[N]*z)
            # print(z, np.cos(theta)/np.sqrt(self.eps_inp)*np.exp(-l_free[N]*z))
            # print(Em_z, self.r)
            Dxm_z = self.eps_inp*Em_z
            Ezm_z = -1/self.eps_out*self.qm[:,None]*Hm_z
        elif z>z_int[-1]:
            # print(z, z_int[-1])
            Hm_z = np.exp(-l_free*(z-z_int[-1]))[:,None]*self.t
            Dxm_z = 1j*(km*np.exp(-l_free*(z-z_int[-1])))[:,None]*self.t
            Ezm_z = -1/self.eps_out*self.qm[:,None]*Hm_z
        else:
            # print('z_int vn  ',z_int*1e7)

            L_idx = get_layer_id(z, z_int)
            Layer = self.Layers[L_idx]

            # _eps = Layer.eps
            # k0_theta = np.sqrt(k0**2-self.qm[N]**2)
            z0=z_int[L_idx]
            z1=z_int[L_idx+1]
            l = k0*Layer.L #l = l_arr[L_idx]
            
            #print(Layer.L)
            A_plus = np.exp(l*(z0-z))
            A_minus = np.exp(l*(z-z1))
            Lam_plus = A_plus[:,None]*Layer.C_plus
            Lam_minus = A_minus[:,None]*Layer.C_minus
            LC_Ex = Lam_plus - Lam_minus
            LC_H = Lam_plus + Lam_minus
            #Em_z = 1j*Layer.V@LC_Ex
            if Layer.plate:
                Hm_z = LC_H
                Dxm_z = 1j*Layer.L[:,None]*LC_Ex
                Ezm_z = -self.qm[:,None]*Hm_z/Layer._eps1
                # print(Ezm_z)
            else:
                Hm_z = Layer.W@LC_H
                Dxm_z = Layer.Dx_coef@LC_Ex
                Ezm_z = Layer.Ez_coef@Hm_z
        #Em_z = np.squeeze(Em_z)
        Hm_z = np.squeeze(Hm_z)
        Ezm_z = np.squeeze(Ezm_z)
        Dxm_z = np.squeeze(Dxm_z)
        return Hm_z, Dxm_z, Ezm_z

    def get_Exz_coefs_on_z(self, z,L_idx, k0, km, l_free,l_arr, N, z_int, theta=0):
         
        if z<0:
            Hm_z = np.exp(l_free*z)[:,None]*self.r
            Hm_z[N] += np.exp(-l_free[N]*z)
            Em_z = -1j*(km*np.exp(l_free*z))[:,None]*self.r/self.eps_inp
            Em_z[N] += np.cos(theta)/np.sqrt(self.eps_inp)*np.exp(-l_free[N]*z)
            # print(z, np.cos(theta)/np.sqrt(self.eps_inp)*np.exp(-l_free[N]*z))
            # print(Em_z, self.r)
            Dxm_z = self.eps_inp*Em_z
            Ezm_z = -1/self.eps_out*self.qm[:,None]*Hm_z
        elif z>z_int[-1]:
            # print(z, z_int[-1])
            Hm_z = np.exp(-l_free*(z-z_int[-1]))[:,None]*self.t
            Dxm_z = 1j*(km*np.exp(-l_free*(z-z_int[-1])))[:,None]*self.t
            Ezm_z = -1/self.eps_out*self.qm[:,None]*Hm_z
        else:
            # print('z_int vn  ',z_int*1e7)

            # L_idx = get_layer_id(z, z_int)
            Layer = self.Layers[L_idx]

            # _eps = Layer.eps
            # k0_theta = np.sqrt(k0**2-self.qm[N]**2)
            z0=z_int[L_idx]
            z1=z_int[L_idx+1]
            # l = k0_theta*Layer.L #    
            l = l_arr[L_idx] # l = k0*Layer.L
            
            #print(Layer.L)
            A_plus = np.exp(l*(z0-z))
            A_minus = np.exp(l*(z-z1))
            Lam_plus = A_plus[:,None]*Layer.C_plus
            Lam_minus = A_minus[:,None]*Layer.C_minus
            LC_Ex = Lam_plus - Lam_minus
            LC_H = Lam_plus + Lam_minus
            #Em_z = 1j*Layer.V@LC_Ex
            if Layer.plate:
                Hm_z = LC_H
                Dxm_z = 1j*Layer.L[:,None]*LC_Ex
                Ezm_z = -self.qm[:,None]*Hm_z/Layer._eps1
                # print(Ezm_z)
            else:
                Hm_z = Layer.W@LC_H
                Dxm_z = Layer.Dx_coef@LC_Ex
                Ezm_z = Layer.Ez_coef@Hm_z
        
        return  Dxm_z[:,0], Ezm_z[:,0]



    def _fields_coefs_on_z(self, z, v, N, Theta=0):
        #NOT USED
        k0 = 2*pi*v/c_light
        tic = time.perf_counter()
        t, r = self.solver(k0,N, field = True, Theta=Theta)
        toc = time.perf_counter()
        T = np.abs(t[N])**2
        R = np.abs(r[N])**2
        #print(f'Time for RSWA solution - {toc-tic} s')
        self.t = t
        self.r = r
        k0_inc =  np.sqrt(self.eps_inp)*sin(Theta*pi/180)
        period = self.Layers[0].period
        qm = k0_inc + qm_fill(period, N, k0)
        self.qm = qm
        km = np.diag(km_fill(qm, self.eps_out))
        z_int = np.zeros(len(self.Layers)+1)

        for idx,Layer in enumerate(self.Layers):
            z_int[idx+1]+= z_int[idx]+Layer.depth
            Layer.Dx_coef = 1j*Layer.W*Layer.L
            Layer.Ez_coef = -Layer.eps_mm*self.qm

        self.z_int = z_int

        l_free =k0*np.diag(km_fill(qm, self.eps_out))
        if z<0:
            Hm_z = np.exp(l_free*z)[:,None]*self.r
            Hm_z[N] += np.exp(-l_free[N]*z)
            Em_z = -1j*(km*np.exp(l_free*z))[:,None]*self.r/self.eps_inp
            Em_z[N] += np.cos(Theta)/np.sqrt(self.eps_inp)*np.exp(-l_free[N]*z)
            Dxm_z = self.eps_inp*Em_z
            Ezm_z =-1/self.eps_out*self.qm[:,None]*Hm_z
        elif z>z_int[-1]:
            # print(z, z_int[-1])
            Hm_z = np.exp(-l_free*(z-z_int[-1]))[:,None]*self.t
            Dxm_z = 1j*(km*np.exp(-l_free*(z-z_int[-1])))[:,None]*self.t
            Ezm_z = -1/self.eps_out*self.qm[:,None]*Hm_z
        else:
            # print('z_int vn  ',z_int*1e7)
 
            L_idx = get_layer_id(z, z_int)
            Layer = self.Layers[L_idx]
            z0=z_int[L_idx]
            z1=z_int[L_idx+1]
            l = k0*Layer.L
            A_plus = np.exp(l*(z0-z))
            A_minus = np.exp(l*(z-z1))
            Lam_plus = A_plus[:,None]*Layer.C_plus
            Lam_minus = A_minus[:,None]*Layer.C_minus
            LC_Ex = Lam_plus - Lam_minus
            LC_H = Lam_plus + Lam_minus
            #Em_z = 1j*Layer.V@LC_Ex
            if Layer.plate:
                # pdb.set_trace()
                Hm_z = LC_H
                Dxm_z = 1j*Layer.L[:,None]*LC_Ex
                Ezm_z = -self.qm[:,None]*Hm_z/Layer._eps1
            else:
                Hm_z = Layer.W@LC_H
                Dxm_z = Layer.Dx_coef@LC_Ex
                Ezm_z = Layer.Ez_coef@Hm_z
        #Em_z = np.squeeze(Em_z)
        Hm_z = np.squeeze(Hm_z)
        Ezm_z = np.squeeze(Ezm_z)
        Dxm_z = np.squeeze(Dxm_z)
        return Hm_z, Dxm_z, Ezm_z

    def field_mapping(self, v, N,Nx, Nz, Zmin, Zmax, Theta=0):
        k0 = 2*pi*v/c_light
        
        tic = time.perf_counter()
        t, r = self.solver(k0,N, Theta=Theta, field = True)
        T_diff, R_diff = self.get_diffraction_orders(v, N, t, r, Theta=Theta)

        self.T_diffraction = T_diff
        self.R_diffraction = R_diff
        toc = time.perf_counter()
        T = np.abs(t[N])**2
        R = np.abs(r[N])**2
        #print(f'Time for RSWA solution - {toc-tic} s')
        self.t = t
        self.r = r
        k0_inc =  np.sqrt(self.eps_inp)*sin(Theta*pi/180)
        period = self.Layers[0].period
        qm = k0_inc + qm_fill(period, N, k0)
        self.qm = qm
        km = np.diag(km_fill(qm, self.eps_out))
        z_int = np.zeros(len(self.Layers)+1)

        for idx,Layer in enumerate(self.Layers):
            z_int[idx+1]+= z_int[idx]+Layer.depth
            Layer.Dx_coef = 1j*Layer.W*Layer.L
            Layer.Ez_coef = -Layer.eps_mm*self.qm
            Layer.l = k0*Layer.L

        self.z_int = z_int
        x_max = period
        Z = np.linspace(Zmin*1e-7, Zmax*1e-7, Nz)
        X = np.linspace(0, x_max, Nx)
        X0 = np.linspace(0, 1, Nx)
        Z_idxs = np.arange(Nz)
        #Z_masks = [(Z>z_l) & (Z<z_h) for z_l, z_h in pairwise(z_int)]
        
        #Z_chanks = [Z_idxs[mask] for mask in Z_masks]
#        print(Z_chanks)
        xx, zz = np.meshgrid(X,Z)
#        _E = np.zeros(shape = xx.shape, dtype = np.complex)
        _Dx = np.zeros(shape = xx.shape, dtype = np.complex128)
        _Ez = np.zeros(shape = xx.shape, dtype = np.complex128)
        _H = np.zeros(shape = xx.shape, dtype = np.complex128)
        X1 = X-self.Layers[0].fill*period/2
        exmX = np.exp(1j*qm[:,None]*k0*X1)
        l_free =k0*np.diag(km_fill(qm, self.eps_out))
        
        for idx_z in Z_idxs:
            z = Z[idx_z]
            Hm_z, Dxm_z, Ezm_z = self.get_field_coefs_on_z(z, k0, km, l_free, N, z_int, theta=Theta)
            _H[idx_z] = Hm_z.dot(exmX)
            _Ez[idx_z] = Ezm_z.dot(exmX)
            _Dx[idx_z] = Dxm_z.dot(exmX)   

        xx0, zz = np.meshgrid(X*1e4,Z)

        _eps = self.eps_structure(xx0,zz, v)
        self._Dx_cache = _Dx
        _Ex = _Dx/_eps
        return xx, zz, _H, _Ex, _Ez, T, R
    

    def Exz_field_mapping(self, v, N,Nx, Nz, Zmin, Zmax, Theta=0):
        # import pdb
        k0 = 2*pi*v/c_light
        tic = time.perf_counter()
        t, r = self.solver(k0,N, Theta=Theta, field = True)
        T_diff, R_diff = self.get_diffraction_orders(v, N, t, r, Theta=Theta)

        self.T_diffraction = T_diff
        self.R_diffraction = R_diff
        toc = time.perf_counter()
        T = np.abs(t[N])**2
        R = np.abs(r[N])**2
        #print(f'Time for RSWA solution - {toc-tic} s')
        self.t = t
        self.r = r
        k0_inc =  np.sqrt(self.eps_inp)*sin(Theta*pi/180)
        period = self.Layers[0].period
        qm = k0_inc + qm_fill(period, N, k0)
        self.qm = qm
        km = np.diag(km_fill(qm, self.eps_out))
        z_int = np.zeros(len(self.Layers)+1)
        l_arr = np.zeros((len(self.Layers), len(self.Layers[0].L)), dtype=np.complex128)

        for idx,Layer in enumerate(self.Layers):
            z_int[idx+1]+= z_int[idx]+Layer.depth
            Layer.Dx_coef = 1j*Layer.W*Layer.L
            Layer.Ez_coef = -Layer.eps_mm*self.qm
            l_arr[idx] = k0*Layer.L

        self.z_int = z_int
        x_max = period
        Z = np.linspace(Zmin*1e-7, Zmax*1e-7, Nz)
        
        X = np.linspace(0, x_max, Nx)
        X0 = np.linspace(0, 1, Nx)
        Z_idxs = np.arange(Nz)

        xx, zz = np.meshgrid(X,Z)
        _Dx = np.zeros(shape = xx.shape, dtype = np.complex128)
        _Ez = np.zeros(shape = xx.shape, dtype = np.complex128)
        X1 = X-self.Layers[0].fill*period/2
        exmX = np.exp(1j*qm[:,None]*k0*X1)
        l_free =k0*np.diag(km_fill(qm, self.eps_out))
        # clip: a sample landing exactly on the last interface would
        # otherwise index one past the final layer (finding H6)
        L_idxs = np.clip(np.searchsorted(z_int, Z, side='right') - 1,
                         0, len(self.Layers) - 1)
        # pdb.set_trace()
        for idx_z in Z_idxs:
            z = Z[idx_z]
            L_idx = L_idxs[idx_z]
            Dxm_z, Ezm_z = self.get_Exz_coefs_on_z(z,L_idx, k0, km, l_free, l_arr, N, z_int, theta=Theta)
            # Ezm_z = adjust_fourier_coeffs(Ezm_z, dx=self.Layers[0].fill/2-0.5)
            # Dxm_z = adjust_fourier_coeffs(Ezm_z, dx=self.Layers[0].fill/2-0.5)
            # _Ez[idx_z] = ifft(2*N*Ezm_z) 
            _Ez[idx_z] = Ezm_z.dot(exmX) 
            #_Dx[idx_z] = ifft(2*N*Dxm_z) 
            _Dx[idx_z] = Dxm_z.dot(exmX)  
        # plt.plot(np.real(tr_Ez)
        #          )
        # plt.plot(np.real(_Ez[-1]))
        xx0, zz = np.meshgrid(X*1e4,Z)

        _eps = self.eps_structure(xx0,zz, v)
        self._Dx_cache = _Dx
        _Ex = _Dx/_eps
        return xx, zz, _Ex, _Ez, T, R


    def electric_field_mapping(self, v, N,Nx, Nz, Zmin, Zmax, Theta=0):
        k0 = 2*pi*v/c_light
        tic = time.perf_counter()
        t, r = self.solver(k0,N, field = True, Theta=Theta)
        toc = time.perf_counter()
        T = np.abs(t[N])**2
        R = np.abs(r[N])**2
        #print(f'Time for RSWA solution - {toc-tic} s')
        self.t = t
        self.r = r
        k0_inc =  np.sqrt(self.eps_inp)*sin(Theta*pi/180)
        period = self.Layers[0].period
        qm = k0_inc + qm_fill(period, N, k0)
        self.qm = qm
        km = np.diag(km_fill(qm, self.eps_out))
        z_int = np.zeros(len(self.Layers)+1)

        for idx,Layer in enumerate(self.Layers):
            z_int[idx+1]+= z_int[idx]+Layer.depth
        self.z_int = z_int
        x_max = period
        Z = np.linspace(Zmin*1e-7, Zmax*1e-7, Nz)
        X = np.linspace(0, x_max, Nx)
        X0 = np.linspace(0, 1, Nx)
        Z_idxs = np.arange(Nz)
        #Z_masks = [(Z>z_l) & (Z<z_h) for z_l, z_h in pairwise(z_int)]
        
        #Z_chanks = [Z_idxs[mask] for mask in Z_masks]
#        print(Z_chanks)
        xx, zz = np.meshgrid(X,Z)
#        _E = np.zeros(shape = xx.shape, dtype = np.complex128)
        _Dx = np.zeros(shape = xx.shape, dtype = np.complex128)
        _Ez = np.zeros(shape = xx.shape, dtype = np.complex128)
        X1 = X-self.Layers[0].fill*period/2
        exmX = np.exp(1j*qm[:,None]*k0*X1)
        l_free =k0*np.diag(km_fill(qm, self.eps_out))
        for idx_z in Z_idxs:
            z = Z[idx_z]
            Hm_z, Dxm_z, Ezm_z = self.get_field_coefs_on_z(z, k0, km, l_free, N, z_int, theta=Theta)
            _Ez[idx_z] = Ezm_z.dot(exmX)
            _Dx[idx_z] = Dxm_z.dot(exmX)   
        xx0, zz = np.meshgrid(X*1e4,Z)
        _eps = self.eps_structure(xx0,zz, v)
        _Ex = _Dx/_eps
        return xx, zz,_Ex, _Ez, T, R

    def _x_cell_weights(self, Nx):
        """Fourier-frame abscissa and cell width for the periodic x grid.

        The field grid is X = linspace(0, period, Nx), whose first and last
        samples are the same physical point.  Dropping the duplicate leaves a
        uniform periodic grid of Nx-1 cells of width 1/(Nx-1); summing
        f_i * h over the samples of a region is exactly conservative, i.e. the
        per-medium integrals of one layer always add up to the full period.
        """
        x_arr = np.linspace(0, 1, Nx)[:-1]
        return fourier_frame_x(x_arr, self.x_shift), 1.0/(Nx - 1)

    def _absorption_from_orders(self, suffix=''):
        """Absorbed fraction summed over every propagating diffraction order.

        Two different quantities are stored by the spectrum drivers and they
        must not be confused:

          spectrT / spectrR   zero-order diffraction EFFICIENCIES
          spectrA             1 - spectrT - spectrR, the complement of the
                              zero order.  This is NOT the absorption once
                              higher orders propagate -- which happens easily
                              for a substrate with eps_out > 1, since orders
                              transmit whenever |q_m| < sqrt(eps_out).
          spectrTfull / spectrRfull   efficiency of every order
          spectrA_full        1 - sum(T_m) - sum(R_m), the true absorption

        The name `spectrA` is kept as it was so that existing scripts and
        figures are unaffected; `spectrA_full` is the quantity to plot as
        absorptivity.
        """
        T = np.real(getattr(self, 'spectrTfull'+suffix))
        R = np.real(getattr(self, 'spectrRfull'+suffix))
        return 1.0 - T.sum(axis=1) - R.sum(axis=1)

    def calculate_partial_loss(self, zz, Ex, Ez, v, Theta, simps_rule=True):
        """Absorbed power fraction of each layer.

        The x integration follows that layer's own eps_segments(), so every
        medium is weighted by its own Im(eps) -- including the core of a
        GratingLayer2, which the previous two-medium split ignored -- and a
        strip that wraps across the period edge is handled correctly.
        """
        z_arr = zz[:, 0]
        mask = (z_arr >= 0) & (z_arr <= self.z_int[-1])
        z_arr = z_arr[mask]
        w = np.abs(Ex[mask])**2 + np.abs(Ez[mask])**2

        xn, dx = self._x_cell_weights(w.shape[1])
        w = w[:, :-1]                       # drop the duplicated period edge

        l_idxs = np.array([get_layer_id(z, self.z_int) for z in z_arr],
                          dtype=np.int32)
        part_losses = np.zeros(len(self.Layers))

        for L_idx, Layer in enumerate(self.Layers):
            zmask = l_idxs == L_idx
            if np.sum(zmask) < 2:
                continue
            _Z = z_arr[zmask] - z_arr[zmask][0]
            _w = w[zmask]
            loss_z = np.zeros(len(_Z))
            for x0, x1, eps_s in Layer.eps_segments(v):
                im_eps = np.imag(eps_s)
                if x1 <= x0 or im_eps == 0:
                    continue
                xmask = (xn >= x0) & (xn < x1)
                if not xmask.any():
                    continue
                loss_z += im_eps*_w[:, xmask].sum(axis=1)*dx
            if simps_rule:
                part_losses[L_idx] = simps(loss_z, _Z)
            else:
                part_losses[L_idx] = CubicSpline(_Z, loss_z).integrate(0, _Z[-1])

        part_losses *= 2*pi*v/(c_light*np.cos(np.radians(Theta)))
        return part_losses

    def get_loss_profile_z(self, zz, Ex, Ez, v, simps_rule = False):
        """Depth profile of the absorbed power density, per unit z.

        Uses the same per-layer segment description as calculate_partial_loss,
        so the two always agree: integrating this profile over z reproduces
        sum(p_losses) at normal incidence.  `simps_rule` is kept for backward
        compatibility and no longer affects the x quadrature.
        """
        z_arr = zz[:,0]
        mask = (z_arr>=0) & (z_arr<=self.z_int[-1])
        z_arr = z_arr[mask]
        w = np.abs(Ex[mask])**2 + np.abs(Ez[mask])**2

        xn, dx = self._x_cell_weights(w.shape[1])
        w = w[:, :-1]
        total_loss_z = np.zeros(len(z_arr))
        for idx, z in enumerate(z_arr):
            Layer = self.Layers[get_layer_id(z, self.z_int)]
            for x0, x1, eps_s in Layer.eps_segments(v):
                im_eps = np.imag(eps_s)
                if x1 <= x0 or im_eps == 0:
                    continue
                xmask = (xn >= x0) & (xn < x1)
                if not xmask.any():
                    continue
                total_loss_z[idx] += im_eps*w[idx, xmask].sum()*dx
        total_loss_z *= 2*pi*v/c_light
        return z_arr, total_loss_z



    def calcTRLandPartLoss(self,v_range, Neq, pxy, z_range, Theta = 0, simps_rule = False, verbose = True, optimized = True):
        N = len(v_range)        
        Nx,Ny = pxy
        z_min, z_max = z_range
        self.vrange = v_range
        self.spectrT = np.zeros(N)
        self.spectrR = np.zeros(N)
        self.spectrTfull = np.zeros((N, 2*Neq+1))
        self.spectrRfull = np.zeros((N, 2*Neq+1))
        self.p_losses = np.zeros((N, len(self.Layers)))
        with progress_bar(N, verbose) as bar:
            for i in range(N):
                v = self.vrange[i]
                maping =self.field_mapping(v, Neq, Nx,Ny, z_min, z_max, Theta=Theta)
                XX, YY, H, Ex, Ez, T, R = maping
                self.p_losses[i] = self.calculate_partial_loss(YY, Ex, Ez, v,Theta, simps_rule = simps_rule)
                self.spectrTfull[i] = self.T_diffraction
                self.spectrRfull[i] = self.R_diffraction
                self.spectrT[i] = self.T_diffraction[Neq]
                self.spectrR[i] = self.R_diffraction[Neq]
                bar.update(i)
        self.spectrA = 1-self.spectrT - self.spectrR
        self.spectrA_full = self._absorption_from_orders()

    def calcTRLandPartLoss_wavelength(self, 
                                      lambda_range, 
                                      Neq, 
                                      pxy, 
                                      z_range, 
                                      Theta=0.0, 
                                      simps_rule=True, 
                                      verbose=True, 
                                      optimized=True, 
                                      units='nm'):
        """
        Calculate Transmission, Reflection, and Partial Loss spectra based on wavelength.

        Args:
            lambda_range (array-like): Array of wavelengths to evaluate.
            Neq (int): Number of equations/modes.
            pxy (tuple): Tuple of (Nx, Ny) grid points.
            z_range (tuple): Tuple of (z_min, z_max) defining the z-coordinate range.
            Theta (float, optional): Incident angle in degrees. Defaults to 0.
            simps_rule (bool, optional): Whether to use Simpson's rule for integration. Defaults to False.
            verbose (bool, optional): If True, prints progress. Defaults to True.
            optimized (bool, optional): If True, use optimized calculation paths if available. Defaults to True.
            units (str, optional): Units of wavelength ('cm', 'm', 'nm', 'um'). Defaults to 'nm'.
        """
        # Convert wavelength range to cm (Gaussian units) based on units
        unit_factors = {'cm': 1, 'm': 100, 'nm': 1e-7, 'um': 1e-4}
        if units not in unit_factors:
            raise ValueError("Unsupported units. Choose from 'cm', 'm', 'nm', 'um'.")
        factor = unit_factors[units]

        # Convert input wavelengths to cm
        self.lambda_range = np.array(lambda_range) * factor
        self.lambda_range_mkm = self.lambda_range*1.0e4  # cm -> um
        N = len(self.lambda_range)

        self.vrange = c_light / self.lambda_range  # cm/s / cm = Hz
        
        Nx, Ny = pxy
        z_min, z_max = z_range
        if verbose:
            print(f'N points = {N}')
        self.spectrT = np.zeros(N)
        self.spectrR = np.zeros(N)
        
        self.spectrTfull = np.zeros((N, 2 * Neq + 1))
        self.spectrRfull = np.zeros((N, 2 * Neq + 1))
        self.p_losses = np.zeros((N, len(self.Layers)))

        if verbose:
            print(f'N_eq = {Neq}')
        bar = progress_bar(N, verbose).start()

        for i in range(N):
            wavelength = self.lambda_range[i]
            frequency = c_light / wavelength  # cm/s / cm = Hz

            # Field mapping
            if optimized:
                # Implement any optimized field mapping here
                # For demonstration, using the standard field_mapping
                maping = self.Exz_field_mapping(frequency, Neq, Nx, Ny, z_min, z_max, Theta=Theta)
            else:
                maping = self.Exz_field_mapping(frequency, Neq, Nx, Ny, z_min, z_max, Theta=Theta)
            
            XX, YY, Ex, Ez, T, R = maping
            
            # Calculate partial losses
            p_loss = self.calculate_partial_loss(YY, Ex, Ez, frequency, Theta=Theta, simps_rule=simps_rule)
            self.p_losses[i] = p_loss
            
            # Store Transmission and Reflection
            self.spectrTfull[i] = self.T_diffraction  # Ensure self.T_diffraction is updated in field_mapping
            self.spectrRfull[i] = self.R_diffraction  # Ensure self.R_diffraction is updated in field_mapping
            self.spectrT[i] = self.T_diffraction[Neq]
            self.spectrR[i] = self.R_diffraction[Neq]
            bar.update(i)
        bar.finish()

        self.spectrA = 1 - self.spectrT - self.spectrR
        self.spectrA_full = self._absorption_from_orders()



    def calcTRL_PartLoss_on_theta(self, theta_range, v, Neq, pxy, z_range, simps_rule = False, verbose = True):
        """Calculate TRL spectra and partial losses for a range of angles for a given frequency."""
        N= len(theta_range)
        Nx,Ny = pxy
        z_min, z_max = z_range
        self.theta_range = theta_range
        self.spectrT_theta = np.zeros(N)
        self.spectrR_theta = np.zeros(N)
        self.spectrTfull_theta = np.zeros((N, 2*Neq+1))
        self.spectrRfull_theta = np.zeros((N, 2*Neq+1))
        self.p_losses_theta = np.zeros((N, len(self.Layers)))
        with progress_bar(N, verbose) as bar:
            for i in range(N):
                Theta = theta_range[i]
                maping =self.field_mapping(v, Neq, Nx,Ny, z_min, z_max, Theta=Theta)
                XX, YY, H, Ex, Ez, T, R = maping
                self.p_losses_theta[i] = self.calculate_partial_loss(YY, Ex, Ez, v,Theta, simps_rule = simps_rule)
                self.spectrTfull_theta[i] = self.T_diffraction
                self.spectrRfull_theta[i] = self.R_diffraction
                self.spectrT_theta[i] = self.T_diffraction[Neq]
                self.spectrR_theta[i] = self.R_diffraction[Neq]
                bar.update(i)
        self.spectrA_theta_full = self._absorption_from_orders('_theta')

    def calcTRLpart_loss_angle_averaged(self,v_range, Neq, pxy, z_range, Theta = 0, N_theta=5, theta_sigma = 1,simps_rule = False, verbose = True):
        """Calculate TRL spectra and partial losses for a range of frequencies with averaging over angles.
        Use parameter theta_sigma to set the averaging angle range.
        Kernel function for averaging is a Gaussian function with sigma = theta_sigma.
        Calculate values for N_theta angles from -theta_sigma to theta_sigma then spline interpolate the results.
        Write the averaged results to the object attributes.
        
        Input parameters:
            v_range - frequency range
            Neq - number of harmonics in the expansion
            pxy - meshgrid size
            z_range - z range for field mapping in nm
            Theta - angle of incidence, default = 0
            N_theta - number of angles for averaging, default = 5
            theta_sigma - sigma for the averaging kernel, default = 1
            simps_rule - use Simpson's rule for integration, default = False
            verbose - print progress bar, default = True
        
        Example:
            struct.calcTRLpart_loss_angle_averaged(v_range, Neq, pxy, z_range, Theta = 10, N_theta=5, theta_sigma = 1,simps_rule = False, verbose = True)
        """
        theta_range = np.linspace(-2*theta_sigma,2*theta_sigma,N_theta)
        norm_factor = np.sqrt(2*np.pi)*theta_sigma*erf(np.sqrt(2))
        kernel_func = lambda theta_x: np.exp(-theta_x**2/(2*theta_sigma**2))/norm_factor
        
        N = len(v_range)        
        Nx,Ny = pxy
        z_min, z_max = z_range
        self.vrange = v_range
        self.spectrT = np.zeros(N)
        self.spectrR = np.zeros(N)
        self.spectrTfull = np.zeros((N, 2*Neq+1))
        self.spectrRfull = np.zeros((N, 2*Neq+1))
        self.p_losses = np.zeros((N, len(self.Layers)))
        with progress_bar(N, verbose) as bar:
            for i in range(N):
                v = self.vrange[i]
                T_theta = np.zeros(N_theta)
                R_theta = np.zeros(N_theta)
                p_losses = np.zeros((N_theta, len(self.Layers)))
                for j, theta in enumerate(theta_range+Theta):
                    maping =self.field_mapping(v, Neq, Nx,Ny, z_min, z_max, Theta=theta)
                    XX, YY, H, Ex, Ez, T, R = maping
                    T_theta[j] = self.T_diffraction[Neq]
                    R_theta[j] = self.R_diffraction[Neq]
                    p_losses[j] = self.calculate_partial_loss(YY, Ex, Ez, v,theta, simps_rule = simps_rule)
                spl_losses = [CubicSpline(theta_range, p_losses[:,i]) for i in range(len(self.Layers))]
                spl_T = CubicSpline(theta_range, T_theta)
                spl_R = CubicSpline(theta_range, R_theta)
                integrand_T = lambda theta: kernel_func(theta)*spl_T(theta)
                self.spectrT[i] = quad(integrand_T, theta_range[0], theta_range[-1])[0]
                integrand_R = lambda theta: kernel_func(theta)*spl_R(theta)
                self.spectrR[i] = quad(integrand_R, theta_range[0], theta_range[-1])[0]
                for j in range(len(self.Layers)):
                    integrand_loss = lambda theta: kernel_func(theta)*spl_losses[j](theta)
                    self.p_losses[i,j] = quad(integrand_loss, theta_range[0], theta_range[-1])[0]
                bar.update(i)

    def transmission(self,v,N):
        # print('N=',N)
        k0 = 2*pi*v/c_light
        t,r = self.solver(k0,N)
        return np.abs(t[N])**2
 
    def get_diffraction_orders(self,v, N, t, r, polar='TM', Theta = 0):
        """ Calculate diffraction orders from the raw fourier field harmonics

        Args:
            v (float): the frequency in hz
            N (int): N retained harmonics in the expansion
            t (complex array): complex foruier harmonics of transmitted field
            r (complex array): complex foruier harmonics of reflected field
            polar (str, optional): polarization flag. Defaults to 'TM'.
            Theta (int, optional): angle of incidence. Defaults to 0.

        Returns:
            tuple of 2 arrays: transmission and reflection difraction orders
        """
        k0 = 2*pi*v/c_light
        
        theta=Theta*pi/180
        
        k0_inc =  np.sqrt(self.eps_inp)*sin(theta)
        period = self.Layers[0].period
        qm = k0_inc + qm_fill(period, N, k0)
        t = np.squeeze(t)
        r = np.squeeze(r)
        if polar=='TM':
            T = np.sqrt(self.eps_inp)*np.sqrt(self.eps_out-qm**2)*np.abs(t)**2/(self.eps_out*np.cos(theta)) 
        else:
            T = np.sqrt(self.eps_out-qm**2)*np.abs(t)**2/(np.sqrt(self.eps_inp)*np.cos(theta)) 
        # normalise by the incident-medium admittance sqrt(eps_inp), not eps_inp
        R = np.sqrt(self.eps_inp-qm**2)*np.abs(r)**2/(np.sqrt(self.eps_inp)*np.cos(theta))
        # print(T)
        mask_T = np.real(self.eps_out-qm**2)>0
        T = np.where(mask_T, T, 0)
        mask_R = np.real(self.eps_inp-qm**2)>0
        R = np.where(mask_R, R, 0)
        return np.real(T), np.real(R)

    def psi(self,v,N, Theta=0, units = 'grad'):
        k0 = 2*pi*v/c_light
        theta=Theta*pi/180
        k0_inc =  np.sqrt(self.eps_inp)*sin(theta)
        period = self.Layers[0].period
        _,r_TM = self.solver(k0,N, Theta=Theta, 
                                polarization="TM",
                                field=False)
        r_TM = np.abs(np.squeeze(r_TM))[N]

        _,r_TE = self.solver(k0,N, Theta=Theta, 
                        polarization="TE",
                        field=False)
        r_TE = np.abs(np.squeeze(r_TE))[N]
        _psi = np.arctan(r_TM/r_TE)
        if units=='grad':
            _psi = np.degrees(_psi)
        return _psi, r_TM, r_TE

    def psi_spectra(self, vrange, Neq=50, vebrose=True, Theta=0, verbose=None):
        # `vebrose` is the historical (misspelled) name; `verbose` wins if given (M16)
        if verbose is not None:
            vebrose = verbose
        self.psi_arr = np.zeros_like(vrange)
        self.r_p = np.zeros_like(vrange)
        self.r_s = np.zeros_like(vrange)
        N = len(vrange)
        self.vrange = vrange
        with progress_bar(N, vebrose) as bar:
            for i in range(N):
                v = self.vrange[i]
                _psi, rp, rs = self.psi(v, Neq, Theta=Theta)
                self.psi_arr[i] = _psi
                self.r_p[i] = rp
                self.r_s[i] = rs
                bar.update(i)

    def TR_full(self,v,N,progres=None, field = False, Theta=0, polar='TM'):
        k0 = 2*pi*v/c_light
        try:
            theta=Theta*pi/180
        except Exception as e:
            print(e)
            print(Theta)

        k0_inc =  np.sqrt(self.eps_inp)*sin(theta)
        period = self.Layers[0].period
        qm = k0_inc + qm_fill(period, N, k0) 
        t,r = self.solver(k0,N, Theta=Theta, 
                                polarization=polar,
                                field=field)
        t = np.squeeze(t)
        # print(t)
        r = np.squeeze(r)
        # print(r)
        if polar=='TM':
            T = np.sqrt(self.eps_inp)*np.sqrt(self.eps_out-qm**2)*np.abs(t)**2/(self.eps_out*np.cos(theta)) 
        else:
            T = np.sqrt(self.eps_out-qm**2)*np.abs(t)**2/(np.sqrt(self.eps_inp)*np.cos(theta)) 
        # normalise by the incident-medium admittance sqrt(eps_inp), not eps_inp
        R = np.sqrt(self.eps_inp-qm**2)*np.abs(r)**2/(np.sqrt(self.eps_inp)*np.cos(theta))
        # print(T)
        mask_T = np.real(self.eps_out-qm**2)>0
        T = np.where(mask_T, T, 0)
        mask_R = np.real(self.eps_inp-qm**2)>0
        R = np.where(mask_R, R, 0)
        return np.real(T), np.real(R)

    def TR(self,v,N, Theta=0, polar='TM', field = False):
        # if not progres is None: 
        #     progres.value=v
        k0 = 2*pi*v/c_light
        # print(f'v={v*1e-12}')
        t,r = self.solver(k0,N, Theta=Theta, polarization=polar, field = field)
        return (np.abs(t[N])**2,np.abs(r[N])**2)
    
    def tr(self,v,N, Theta=0, polar='TM', field = False):
        # if not progres is None: 
        #     progres.value=v
        k0 = 2*pi*v/c_light
        # print(f'v={v*1e-12}')
        t,r = self.solver(k0,N, Theta=Theta, polarization=polar, field = field)
        return t[N], r[N]

    def calcTRLspectra(self,v0, v1,vrange = None, Theta =0, N=100,Neq=50,units='Hz',polar='TM',  vebrose = True, verbose = None):
        # `vebrose` is the historical (misspelled) name; `verbose` wins if given (M16)
        if verbose is not None:
            vebrose = verbose
        if vrange is not None:
            self.vrange = vrange
            N = len(vrange)
        else:
            self.vrange = np.linspace(v0,v1,N)
        if vebrose:
            print(f'N points = {N}')
        self.spectrT = np.zeros(N)
        self.spectrR = np.zeros(N)
        
        self.spectrTfull = np.zeros((N, 2*Neq+1))
        self.spectrRfull = np.zeros((N, 2*Neq+1))
        #print(self.spectrTfull.shape)
        if vebrose:
            print(f'N_eq = {Neq}')
        with progress_bar(N, vebrose) as bar:
            for i in range(N):
                v = self.vrange[i]
                T, R = self.TR_full(v, Neq, Theta=Theta, polar = polar, field = False)
                self.spectrT[i] = T[Neq]
                self.spectrR[i] = R[Neq]
                self.spectrTfull[i] = T
                self.spectrRfull[i] = R
                bar.update(i)
        self.spectrA = 1-self.spectrT - self.spectrR
        self.spectrA_full = self._absorption_from_orders()

    def calcTRLspectra_wavelength(self, 
                                lambda0, 
                                lambda1, 
                                lambda_range=None, 
                                Theta=0, 
                                N=100, 
                                Neq=50, 
                                units='nm', 
                                polar='TM',  
                                verbose=True):
        """
        Calculate Transmission, Reflection, and Absorption spectra based on wavelength.

        Args:
            lambda0 (float): Starting wavelength.
            lambda1 (float): Ending wavelength.
            lambda_range (array-like, optional): Specific wavelengths to evaluate. Defaults to None.
            Theta (float, optional): Incident angle in degrees. Defaults to 0.
            N (int, optional): Number of points in the spectrum. Defaults to 100.
            Neq (int, optional): Number of equations/modes. Defaults to 50.
            units (str, optional): Units of wavelength ('nm', 'um', 'm'). Defaults to 'nm'.
            polar (str, optional): Polarization ('TM' or 'TE'). Defaults to 'TM'.
            verbose (bool, optional): If True, prints progress. Defaults to True.
        """
        # Convert wavelength range to meters based on units
        C = c_light*1e-2   # m/s, the same constant used everywhere else
        unit_factors = {'m': 1, 'nm': 1e-9, 'um': 1e-6}
        if units not in unit_factors:
            raise ValueError("Unsupported units. Choose from 'm', 'nm', 'um'.")
        factor = unit_factors[units]
        
        # Define wavelength range in meters
        if lambda_range is not None:
            self.lambda_range = np.array(lambda_range) * factor
            N = len(self.lambda_range)
        else:
            self.lambda_range = np.linspace(lambda0, lambda1, N) * factor
            self.lambda_range_mkm = self.lambda_range*1.0e6
        if verbose:
            print(f'N points = {N}')
        self.spectrT = np.zeros(N)
        self.spectrR = np.zeros(N)
        
        self.spectrTfull = np.zeros((N, 2 * Neq + 1))
        self.spectrRfull = np.zeros((N, 2 * Neq + 1))
        
        if verbose:
            print(f'N_eq = {Neq}')
        bar = progress_bar(N, verbose).start()

        for i in range(N):
            wavelength = self.lambda_range[i]
            frequency = C / wavelength  # Convert wavelength to frequency
            T, R = self.TR_full(frequency, Neq, Theta=Theta, polar=polar, field=False)
            self.spectrT[i] = T[Neq]
            self.spectrR[i] = R[Neq]
            self.spectrTfull[i] = T
            self.spectrRfull[i] = R
            bar.update(i)
        bar.finish()

        self.spectrA = 1 - self.spectrT - self.spectrR
        self.spectrA_full = self._absorption_from_orders()



    def calcTRL_theta(self,theta_range, v, Neq = 50, verbose = True, polar = 'TM'):
        N = len(theta_range)
        self.theta_range = theta_range
        self.spectrT_theta = np.zeros(N)
        self.spectrR_theta = np.zeros(N)
        self.spectrTfull_theta = np.zeros((N, 2*Neq+1))
        self.spectrRfull_theta = np.zeros((N, 2*Neq+1))
        with progress_bar(N, verbose) as bar:
            for i in range(N):
                Theta = theta_range[i]
                T, R = self.TR_full(v, Neq, Theta=Theta, field = False, polar= polar)
                self.spectrT_theta[i] = T[Neq]
                self.spectrR_theta[i] = R[Neq]
                self.spectrTfull_theta[i] = T
                self.spectrRfull_theta[i] = R
                bar.update(i)
        self.spectrA_theta = 1-self.spectrT_theta - self.spectrR_theta
        self.spectrA_theta_full = self._absorption_from_orders('_theta')


    def spectraTMTE(self,v0,v1,N=100,Neq=50,units='THz',mode='T'):
        if units=='mkm':
            vs = l = np.linspace(v0,v1,N) 
            v = c_light/(1e-4*l)
        elif units=='THz':    
            v = np.linspace(v0,v1,N)
            vs = v*1e-12 
        elif units=='cm':
            vs = np.linspace(v0,v1,N)
            v = vs*c_light

        T_v = np.vectorize(lambda x: self.TR(x,Neq,polar='TM'))
        T_TM,R_TM = T_v(v)
        T_v = np.vectorize(lambda x: self.TR(x,Neq,polar='TE'))
        T_TE,R_TE = T_v(v)
        if mode=='T':
            return vs,T_TM,T_TE
        else: 
            return vs,R_TM,R_TE

    def GridPlot(self,dict_params,v0,v1,
        N=100,Neq=50,
        units='THz',mode='T',
        figtitle=None,
        average = False,
        level=0,
        PATH='data/'):
        """
            dict_params,
            v0,v1,
            N=100,
            Neq=50,
            units='THz',mode='T',
            figtitle=None,
            average = False
        """
        back=deepcopy(self)
        names = tuple(dict_params.keys())
        par1 = np.array(dict_params[names[0]])
        par2 = np.array(dict_params[names[1]])
        Nc,Nr= len(par1),len(par2)
        fig,plts = plt.subplots(nrows=Nr,
                                ncols=Nc,
                                # sharex=True,
                                sharey='row',
                                figsize=(3.45*Nc,3.2*Nr))
        X,Y = np.meshgrid(par1,par2)
        figtitle is None or fig.suptitle(figtitle)
        i=0
        for x,y,subplt in zip(X.ravel(),Y.ravel(),plts.ravel()):
            [setattr(layer,names[0],x) for layer in self.Layers if names[0]!='depth_mkm']
            [setattr(layer,names[1],y) for layer in self.Layers if names[1]!='depth_mkm']
            [setattr(self.Layers[0],names[i],(x,y)[i]) for i,name in enumerate(names) if names[i]=='depth_mkm']
            vs,T_TM,T_TE = self.spectraTMTE(v0,v1,N=N, Neq=Neq, units=units)
            if average:
                vs0=vs
                vsm_av,TMav = average_max1(vs0,T_TM, raw=False)
                vse_av,TEav = average_max1(vs0,T_TE, raw=False)
                for _ in range(level):
                    vsm_av,TMav = average8cm(vsm_av,TMav, units=units)
                    vse_av,TEav = average8cm(vse_av,TEav, units=units)
                subplt.plot(vsm_av,TMav, label ='TM (p)')
                subplt.plot(vse_av,TEav, label ='TE (s)')
                savearr([vsm_av,TMav],PATH+repr(self)+'TM.dat')
                savearr([vse_av,TEav],PATH+repr(self)+'TE.dat')
            else:
                subplt.plot(vs,T_TM, label ='TM (p)')
                subplt.plot(vs,T_TE, label ='TE (s)')
            subplt.set_xlim((vs.min(),vs.max()))
            subplt1 = subplt.twiny()
            ls=1e4/vs
            subplt.set_xlim((vs.min(),vs.max()))
            subplt1.set_xlim((ls.min(),ls.max()))
            # step = (ls.max()-ls.min())/5
            X_up = np.linspace(ls.max(),ls.min(),6)
            # X_up = ls[tiks_ind]
            subplt1.set_xticks(X_up)
            f_trans = np.vectorize(lambda s:f'{s:.0f}')
            subplt1.set_xticklabels(f_trans(X_up[::-1]))
            # subplt.set_ylim((0,1))
            if i%Nc == 0:
                subplt.set_ylabel(mode)
            i+=1
            if i > X.size-2:
                # if units=='THz':
                subplt.set_xlabel(units)
                # elif units=='mkm':
                    # subplt.set_xlabel(units)
            subplt.legend(loc='upper left')
            #f'{names[0]} = {x*1e4:3.0f}'+r' $\frac{cm^2}{V\times s}$,'+f' {names[1]} = {y:2.1e}'
            subplt1.set_xlabel(self.title_format({names[0]:x,names[1]:y}), size=9.5)

        fig.subplots_adjust(wspace=0.18, hspace=0.4)
        self.Layers = back.Layers   # restore; rebinding `self` restored nothing

    def title_format(self,dic_args):
        return ', '.join([self.titles[key].format(val) for key,val in dic_args.items()])


