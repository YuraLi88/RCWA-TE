import numpy as np
from math import pi
from scipy.linalg import toeplitz
from scipy.special import j1
from numba import njit
from scipy.integrate import quad 

delta = 1e-5
p = 1.0-delta
@njit
def eps_i(m,f):
	if m==0:
		return f
	else:
		return (p**m)*np.sin(pi*m*f)/(pi*m)

def eps_i_ellipse(m,f):
	if m==0:
		return f
	else:
		return j1(pi*m*f)/(pi*m)


def eps_1(eps0,eps1,fill,N):
	eps = lambda x: eps_i(x,fill)
	eps = np.vectorize(eps)
	f_coefs = (eps1-eps0)*eps(np.arange(-N,N+1))
	f_coefs[N] += eps0
	return f_coefs	

def eps_12(eps0,eps1, eps2,f1, f2, N):
    _eps1 = lambda x: eps_i(x,f1)
    _eps1 = np.vectorize(_eps1)
    _eps2 = lambda x: eps_i(x,f2)
    _eps2 = np.vectorize(_eps2)
    f_coefs = (eps1-eps0)*_eps2(np.arange(-N,N+1))
    f_coefs += (eps2-eps1)*_eps1(np.arange(-N,N+1))
    f_coefs[N] += eps0
    return f_coefs

def eps_inv(eps0,eps1,fill,N):
	eps = lambda x: eps_i(x,fill)
	eps = np.vectorize(eps)
	assert eps0 != 0, 'eps0 must be positive'
	assert eps1 != 0, 'eps1 must be positive'
	eps0=1/eps0
	eps1=1/eps1
	f_coefs = (eps1-eps0)*eps(np.arange(-N,N+1))
	f_coefs[N] += eps0
	return f_coefs	


# print(eps_1(0,1,0.5,5))
@njit
def toeplitz_0(a):
	"""
		Input a: one dimensional np.array size 4*N+1
		Output b: two dimensional toeplitz matrix.
		Using example
		>>> a=np.arange(9)
		>>> a
		array([0, 1, 2, 3, 4, 5, 6, 7, 8])
		>>> toeplitz_0(a)
		array([[4, 5, 6, 7, 8],
		       [3, 4, 5, 6, 7],
		       [2, 3, 4, 5, 6],
		       [1, 2, 3, 4, 5],
		       [0, 1, 2, 3, 4]])
		>>>
	"""
	N=a.shape[0]
	assert N%4==1, 'shape input matrix must be 4*N+1'
	N//=4
	res = np.zeros((2*N+1,2*N+1),dtype=a.dtype)
	for i in range(0,2*N+1):
		res[i] = a[2*N-i:4*N+1-i]
	return res

def toeplitz_1(a):
	N=a.shape[0]
	assert N%4==1, 'shape input matrix must be 4*N+1'
	N//=4
	return toeplitz(a)[:2*N+1,2*N:4*N+1] 

# a=np.arange(-6,7)
# print(a)
# print(a.dtype)
# print(toeplitz_0(a))


# CONTOURInSb

def eps_x_smooth(x, fills, eps_arr, dx = 0.03):
    for idx,fill in enumerate(fills):
        if x<fill/2-dx:
            return eps_arr[idx]
        if x<fill/2:
            n1 = np.sqrt(eps_arr[idx])
            n0 = np.sqrt(eps_arr[idx+1])
            coef = (n1-n0)/dx
            return (n0+coef*(fill/2-x))**2
    return eps_arr[-1]


def get_fourier_coefs(fun,N):
    c_m = np.zeros(N+1, dtype=np.complex128)
    c_m[0] = 2*quad(lambda x:np.real(fun(x)),0,0.5, limit = 100)[0]
    c_m[0] += 2j*quad(lambda x:np.imag(fun(x)),0,0.5, limit = 100)[0]
    for idx in range(1, N+1):
        k = 2*np.pi*idx
        c_m[idx] = 2*quad(lambda x:np.real(fun(x)),0,0.5, limit = 100,weight = 'cos', wvar = k)[0]
        c_m[idx] += 2j*quad(lambda x:np.imag(fun(x)),0,0.5, limit = 100,weight = 'cos', wvar = k)[0]
    full_c = np.zeros(2*N+1, dtype=np.complex128)
    full_c[N:] = c_m
    full_c[:N] = c_m[1:][::-1]
    return full_c

def get_eps_smoothed_coefs(N, fills, eps_arr, dx):
    eps_x = lambda x: eps_x_smooth(x, fills, eps_arr, dx)
    eps_coefs = get_fourier_coefs(eps_x,N)
    return eps_coefs