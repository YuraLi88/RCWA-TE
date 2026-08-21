import re
from itertools import tee
import numpy as np 
from scipy.interpolate import CubicSpline

c_light = 3e10 # cm/s

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def float_parts(x):
    delim=re.compile(r'e\+?')
    return delim.split(f'{x:8.2e}'.strip())

class phys_val():
    """docstring for phys_val"""
    def __init__(self, name, unit):
        self.name = name
        self.unit = unit
    def format(self,value):
        if  1e-2<value<1e4:
            return f"{self.name}={value:4.1f} {self.unit}"
        else:
            digits,ex = float_parts(value)
            if ex[0]=='-':
                ex='-'+str(ex[1:]).lstrip('0')
            else:
                ex=str(ex).lstrip('0')
            r,l = digits.split('.')
            digits = f"{r}.{l[0]+l[1:].strip('0')}"
            my_float = f'{digits}'+r'$\times 10^{'+ex+'}$'
            return f"{self.name}="+my_float+ f"{self.unit}"
        

  

def average8cm(X,Y,units='THz',resolution_cm=16):
    if units=='Hz':
        delta = resolution_cm*c_light
    if units=='THz':
        delta = resolution_cm*c_light*1e-12
    if units=='cm':
        delta = resolution_cm
    dn = X[X<X[0]+delta].shape[0]
    assert dn>1, 'delta is to small!'
    assert delta<(X[-1]-X[0]), 'delta is to large!' 
    Nmax=X.shape[0]-dn
    # print('Nmax=',Nmax)
    X1 = X[:Nmax]
    Y1 = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        Y1[i] = Y[i:i+dn].mean()
    return X1,Y1


def average(X,N=10):
    Y=np.zeros(X.shape[0])
    assert N%2==0, 'N must be even number!'
    head=X[0]*np.ones((N//2))
    tail=X[-1]*np.ones((N//2))
    Xm=np.block([head,X,tail])
    for i in range(Y.shape[0]):
        Y[i]=np.mean(Xm[i:i+N])
    return Y


def findmax(a):
    indices=[0]
    minstep = (a.shape[0]-1)//10
    tol=0e-2
    for i in range(1,a.shape[0]-1):
        if (a[i-1]+tol<a[i]) and (a[i+1]+tol<a[i]):
            indices.append(i)
        if i-indices[-1]>minstep:
            indices.append(i)
    indices.append(a.shape[0]-1)
    return indices

def average_max1(X,Y,raw=False,extr='periodic'):
    indices = findmax(Y)
    N = len(indices)-1
    X1 = np.zeros((N))
    Y1 = np.zeros((N))
    for x in range(N):
        i=indices[x]
        j=indices[x+1]
        Y1[x]=np.mean(Y[i:j])
        X1[x]=np.mean(X[i:j])
    Y1[0]= Y1[1] #np.mean(Y)
    Y1[-1]= Y1[-2]  #np.mean(Y)
    cs=CubicSpline(X1,Y1,extrapolate=extr)
    if raw:
        return X1,Y1
    else:
        return X,cs(X)

# def print_m(a,c=100):
#     print((a*c).astype(np.int32))