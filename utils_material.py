import numpy as np
from matplotlib import pyplot as plt


h_plank = 6.62607015e-34
elCI= 1.602e-19 
c_lightCI = 3e8

def e2lam_mkm(E_mev):
    return (1e9*h_plank)*c_lightCI/(elCI*E_mev)
def lam2e(l):
    return (1e9*h_plank)*c_lightCI/(elCI*l)
def nu2e(v):
    E = 1e3*h_plank*v/elCI
    return E
def Eg_InSb(T):
    E = 0.235-3.2E-4*T**2/(220+T)
    return E

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
        Eg = self.Eg #0.228
        x=hp_eV*v/Eg
        ReEps, ImEps = self.eps0,0.
        if x<1e-4:
            ReEps+=A*(1+5/16*x**2)/4
        elif (x<1):
            ReEps+=A*(2-np.sqrt(1+x)-np.sqrt(1-x))/x**2
        else:
            if x<1e10:
                ImEps+=A*np.sqrt(x-1)/x**2 
                ReEps+=A*(2-np.sqrt(1+x))/x**2
            else:
                ImEps+=A*x**(-3/2) 
                ReEps+=-A*x**(-3/2)

        return ReEps+1j*ImEps

class Lorentz(object):
    """
    A class representing a Lorentz oscillator model used in optics and photonics.

    This model is used to calculate the complex dielectric constant (permittivity) 
    of a material at different frequencies, as well as reflectivity and related 
    quantities. It is based on the Lorentz oscillator model which describes the 
    response of bound electrons to an external electromagnetic field.

    Attributes:
        w0 (float): Resonant frequency of the oscillator in radians per second. 
                    Default is 3e12.
        gamma (float): Damping factor representing the loss in the system. 
                       Default is 2e11.
        f0 (float): Oscillator strength, computed as f0 * gamma * w0. Default 
                    value for f0 is 0.9, but the effective oscillator strength 
                    used is this value multiplied by gamma and w0.
        eps0 (float): Static permittivity (dielectric constant at zero frequency). 
                      Default is 1.

    Methods:
        eps_w(w): Calculates the complex dielectric constant at a given frequency.
            Parameters:
                w (float): Angular frequency at which to compute the permittivity.
            Returns:
                complex: The complex dielectric constant at frequency w.
        
        r_w(w): Calculates the complex reflection coefficient at a given frequency.
            Parameters:
                w (float): Angular frequency at which to compute the reflection 
                           coefficient.
            Returns:
                complex: The complex reflection coefficient at frequency w.
        
        R_w(w): Calculates the reflectivity at a given frequency.
            Parameters:
                w (float): Angular frequency at which to compute the reflectivity.
            Returns:
                float: Reflectivity at frequency w, which is the magnitude squared 
                       of the reflection coefficient.
        
        r_log_w(w): Calculates the natural logarithm of the absolute value of the 
                    reflection coefficient at a given frequency.
            Parameters:
                w (float): Angular frequency at which to compute the log reflection.
            Returns:
                float: The natural logarithm of the absolute value of the reflection 
                       coefficient at frequency w.
    """
    def __init__(self, w0 = 3e12, gamma=2e11, f0=0.9, eps0=1):
        super(Lorentz, self).__init__()
        self.w0 = w0
        self.gamma = gamma
        self.f0 = f0*gamma*w0
        self.eps0 = eps0
    
    def eps_w(self,w):
        eps = self.eps0 + self.f0/((self.w0**2-w**2) - 1j*w*self.gamma)
        return eps

    def r_w(self, w):
        eps = self.eps_w(w)
        r = (np.sqrt(eps)-1)/(np.sqrt(eps)+1)
        return r

    def R_w(self, w):
        r = self.r_w(w)
        return np.abs(r)**2 
    
    def r_log_w(self, w):
        r = self.r_w(w)
        return np.log(np.abs(r))


_InSb = InSb(Eg = 0.223)

epsInSb = _InSb.epsInSb

def main():
    w_range = np.linspace(1e11, 1e13, 1000)
    model = Lorentz()
    eps = np.vectorize(lambda x: model.eps_w(x))
    y  = eps(w_range)
    plt.plot(w_range, np.real(y))
    plt.plot(w_range, np.imag(y))
    plt.show()
    R = model.R_w(w_range)
    plt.plot(w_range, R)
    plt.show()
    plt.plot(w_range, np.log(np.sqrt(R)))
    plt.show()
if __name__ == '__main__':
	main()