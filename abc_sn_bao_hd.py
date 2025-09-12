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


data = pd.read_csv('/Users/alfonsozapata/Documents/SimpleMC/simplemc/data/Pantheon+SH0ES.dat', sep=r'\s+')
zcmb = data['zCMB']



arr_hub = np.loadtxt('/Users/alfonsozapata/Documents/SimpleMC/simplemc/data/Hz_all.dat')
z_obs= arr_hub[:,0]
hub_obs = arr_hub[:,1]
error_obs = arr_hub[:,2]

################################
#  PRIORS
################################

def prior_w0():
    return uniform.rvs(loc=-2.0, scale=1.0)
def prior_wa():
    return uniform.rvs(loc=-1.0, scale=2.0)

################################
#  SIMULATOR OBSERVABLES
################################


def mu_sim(w0,wa): 
    mod = np.array([distance_modulus(z,w0,wa) for z in zcmb])
    return mod

def Hubble_sim(w0, wa):
    try:
        Hz_sim = np.array([hubble_normalized_cpl(z, w0, wa) for z in z_obs])
        # Check for   # for NaN entries 
        if np.any(~np.isfinite(Hz_sim)) or np.any(Hz_sim <= 0):
            return None
        return Hz_sim
  
    except:
        return None
    

################################
#  DISTANCE FUNCTIONS   
###############################

def sn_distance_chi2(mu_obs, _mu_sim,cov):
    """
    (Data-Sim)^T C^-1 (Data-Sim)
    """
    # Load covariance matrix
    #cov = np.loadtxt('/Users/alfonsozapata/Documents/SimpleMC/simplemc/data/Pantheon+SH0ES_STAT+SYS.cov')
    cov_inv = np.linalg.inv(cov)
    # Residuals
    delta = mu_obs - mu_sim_vals
    # Chi^2 argument
    chi2 = np.dot(delta, np.dot(cov_inv, delta))
    return chi2


    
def dist2_hub(observed, errors, simulated):
    if simulated is None:
        return np.inf

    return np.sum((observed - simulated)**2 / errors**2)





###############################
#  REJECTION ABC ALGORITHM
###############################

def rejection_sim_sn(prior_w0, prior_wa, n_accepted, epsilon, n_sims_before_update=100000):
    """
    ABC Rejection Algorithm for SN data using (Data-Sim)^T C^-1 (Data-Sim) distance.
    """
    accepted_particles = []
    total_simulations = 0
    distances = []

    # Load observed SN data and covariance matrix
    mu_obs = data['MU'].values
    cov = np.loadtxt('/Users/alfonsozapata/Documents/SimpleMC/simplemc/data/Pantheon+SH0ES_STAT+SYS.cov')
    cov_inv = np.linalg.inv(cov)

    print(f"Starting ABC rejection for SN. Need to accept {n_accepted} particles.")
    print(f"Tolerance: epsilon = {epsilon}")

    while len(accepted_particles) < n_accepted:
        # Sample from priors
        w0_par = prior_w0()
        wa_par = prior_wa()

        # Simulate SN data
        mu_sim_vals = mu_sim(w0_par, wa_par)

        # Compute chi^2 distance
        delta = mu_obs - mu_sim_vals
        chi2 = np.dot(delta, np.dot(cov_inv, delta))
        distances.append(chi2)
        total_simulations += 1

        # Accept if within tolerance
        if chi2 <= epsilon:
            accepted_particles.append((w0_par, wa_par, chi2))
            if len(accepted_particles) % max(1, n_accepted // 10) == 0:
                print(f"Accepted {len(accepted_particles)}/{n_accepted} particles. "
                      f"Acceptance rate: {len(accepted_particles)/total_simulations:.4f}")

        if total_simulations % n_sims_before_update == 0:
            print(f"Completed {total_simulations} simulations. "
                  f"Accepted: {len(accepted_particles)}. "
                  f"Current acceptance rate: {len(accepted_particles)/total_simulations:.6f}")

    print(f"Completed! Total sim: {total_simulations}. "
          f"Final acceptance rate: {len(accepted_particles)/total_simulations:.6f}")

    accepted_w0 = np.array([p[0] for p in accepted_particles])
    accepted_wa = np.array([p[1] for p in accepted_particles])
    accepted_chi2 = np.array([p[2] for p in accepted_particles])

    return accepted_w0, accepted_wa, accepted_chi2, total_simulations, distances

