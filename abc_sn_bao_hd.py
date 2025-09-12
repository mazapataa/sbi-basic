import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
import pandas as pd
import matplotlib.image as mpimg
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm, uniform

Omega_m0 = 0.3
Omega_L0 = 0.7
h = 0.71
H0 = h *100
zvals = np.linspace(0,3,100)
#####################################
#
#       COSMOLOGICAL DISTANCES 
#
#####################################



def hubble_normalized_cpl(z, w0, wa):
    
    matter_term = Omega_m0 * (1 + z)**3
    
    de_term = Omega_L0 * (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z))
    return  H0 * np.sqrt(matter_term + de_term)

def H():
       return 100*h

def HUBz(z,w0,wa):
    return H()*hubble_normalized_cpl(z,w0,wa)

def inverse_hubble_normalized_cpl(z,w0,wa):

    return 1./hubble_normalized_cpl(z,w0,wa)

def hubble_normalized_a(a,w0,wa):
    return hubble_normalized_cpl(1./a-1,w0,wa)

def hubble_prime_normalized_a(a,w0,wa):
    return derivative(hubble_normalized_a(a,w0,wa), a, dx=1e-6)

def D_H():
    """
    Hubble distance in units of MPc (David Hogg arxiv: astro-ph/9905116v4)
    """
    return 2997.98/h

def comoving_distance_cpl(z,w0,wa):
    return D_H() *quad(inverse_hubble_normalized_cpl(z,w0,wa),0, z,epsabs=1e-6, epsrel=1e-6, limit=200)[0]

# angular diameter distance in units of MPc
def angular_diameter_distance_cpl(z,w0,wa):
    """
    Angular diameter distance as function of redshift z
    in units of MPc
    """
    d_c = comoving_distance_cpl(z,w0,wa)/(1.+z)
    return d_c

# luminosity distance in units of MPc
def luminosity_distance_cpl(z,w0,wa):
    d_c = comoving_distance_cpl(z,w0,wa)
    return (1 + z) * d_c

def build_luminosity_distance_interpolator(w0,wa,zmax=1.5,npoints=150):
    zgrid = np.linspace(1e-4, zmax, npoints)
    dLgrid = np.array([luminosity_distance_cpl(z,w0,wa) for z in zgrid])
    _dL_interp = interp1d(zgrid, dLgrid, kind='cubic', bounds_error=False, fill_value='extrapolate')
    _zmax_interp = zmax

def luminosity_distance_fast(z):
    if hasattr('_dL_interp') and z <= _zmax_interp:
        return _dL_interp(z)
    else:
        return luminosity_distance_cpl(z,w0,wa)

def distance_modulus(z,w0,wa):
    try:
        dL = luminosity_distance_fast(z,w0,wa)
        return 5 * np.log10(dL) + 25
    except Exception as e:
        print(f"[ERROR] distance_modulus failed at z = {z}: {e}")
        return 1e6  # Return large chi2 penalty


###############################
#
#      DATA: PP+SH0ES, DESIR2, CC 
#
##############################


data = pd.read_csv('/Users/alfonsozapata/Documents/SimpleMC/simplemc/data/binned_pantheon.txt', sep=r'\s+')
zcmb = data['zcmb']


arr_hub = np.loadtxt('/Users/alfonsozapata/Documents/SimpleMC/simplemc/data/Hz_all.dat')
z_obs= arr_hub[:,0]
hub_obs = arr_hub[:,1]
error_obs = arr_hub[:,2]

