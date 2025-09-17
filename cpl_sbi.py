import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sbi
from sbi import analysis, utils
from sbi.inference import SNPE
from sbi.utils.user_input_checks import check_prior
from scipy.integrate import quad
from getdist import plots, MCSamples
import pandas as pd
from scipy.interpolate import interp1d
import os

torch.manual_seed(42)
np.random.seed(42)

start_time = time.time()

# --- 1. Define the Cosmological Model (CPL) ---
# The CPL parameterization: w(z) = w0 + wa * z / (1 + z)
# We'll use a simple flat universe: H(z)^2 = H0^2 [ Ω_m(1+z)^3 + Ω_DE(z) ]

# Load real H(z) observational data
try:
    arr_hub = np.loadtxt('/Users/alfonsozapata/Documents/SimpleMC/simplemc/data/Hz_all.dat') 
except FileNotFoundError:
    try:
        arr_hub = np.loadtxt('/home/alfonsozapata/SimpleMC/simplemc/data/Hz_all.dat')
    except FileNotFoundError:
        raise FileNotFoundError("Could not find H(z) data file")

z_obs = arr_hub[:, 0]
hub_obs = arr_hub[:, 1]
error_obs = arr_hub[:, 2]  # Real observational errors
n_hz_observations = len(z_obs)

print(f"Loaded {n_hz_observations} H(z) measurements")
print(f"Redshift range: {z_obs.min():.3f} to {z_obs.max():.3f}")
print(f"H(z) range: {hub_obs.min():.1f} to {hub_obs.max():.1f} km/s/Mpc")
print(f"Error range: {error_obs.min():.1f} to {error_obs.max():.1f} km/s/Mpc")

# Load Pantheon SN data (create mock data if files don't exist)
def create_mock_pantheon_data():
    """Create mock Pantheon data for demonstration"""
    np.random.seed(42)
    z_sn = np.random.uniform(0.01, 2.3, 50)  # 50 mock SNe
    z_sn = np.sort(z_sn)
    
    # Generate mock distance moduli with LCDM + noise
    H0_true, Om_true = 70.0, 0.3
    dL_true = []
    for z in z_sn:
        # Simple LCDM luminosity distance approximation
        dL = (3000/H0_true) * z * (1 + z/2)  # Simplified for demo
        dL_true.append(dL)
    
    dL_true = np.array(dL_true)
    mu_true = 5 * np.log10(dL_true) + 25
    mu_err = 0.15  # Typical SN error
    mu_obs = mu_true + np.random.normal(0, mu_err, len(z_sn))
    
    # Create mock dataframe
    df = pd.DataFrame({
        'zCMB': z_sn,
        'mu': mu_obs,
        'mu_err': np.full_like(z_sn, mu_err)
    })
    
    # Create mock covariance (diagonal for simplicity)
    cov = np.diag(mu_err**2 * np.ones(len(z_sn)))
    
    return df, cov

# Try to load real Pantheon data, otherwise use mock data
try:
    df_sn = pd.read_csv("binned_pantheon.txt", sep=r'\s+')
    if 'mu_err' not in df_sn.columns:
        df_sn['mu_err'] = 0.15  # Add default error if missing
    
    try:
        with open("binned_cov_pantheon.txt") as f:
            lines = f.readlines()
        n_sn = int(lines[0].strip())
        vals = np.array([float(x.strip()) for x in lines[1:]])
        cov_sn = vals.reshape((n_sn, n_sn))
    except FileNotFoundError:
        cov_sn = np.diag(df_sn['mu_err'].values**2)
    
    print("Loaded real Pantheon data")
except FileNotFoundError:
    print("Pantheon data files not found, creating mock data...")
    df_sn, cov_sn = create_mock_pantheon_data()

z_sn = df_sn['zCMB'].values
if 'mu' in df_sn.columns:
    mu_obs_sn = df_sn['mu'].values
else:
    mu_obs_sn = df_sn.iloc[:, 1].values  # Assume second column is mu
mu_err_sn = df_sn['mu_err'].values
n_sn_observations = len(z_sn)

print(f"Loaded {n_sn_observations} SN measurements")
print(f"SN redshift range: {z_sn.min():.3f} to {z_sn.max():.3f}")
print(f"Distance modulus range: {mu_obs_sn.min():.2f} to {mu_obs_sn.max():.2f}")

def hubble_cpl(z, H0, Om0, w0, wa):
    """
    Hubble parameter for flat universe with CPL dark energy.
    """
    matter_term = Om0 * (1 + z)**3
    Ode0 = 1 - Om0 
    de_term = Ode0 * (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z))
    return H0 * np.sqrt(matter_term + de_term)

# --- Cosmological Distance Functions for Pantheon SN ---
def hubble_normalized_cpl(z, H0, Om0, w0, wa):
    """Normalized Hubble parameter E(z) = H(z)/H0 for CPL model"""
    z = np.asarray(z)
    matter_term = Om0 * (1 + z)**3
    OL0 = 1.0 - Om0  # Flat universe
    de_exponent = 3 * (1 + w0 + wa)
    de_evolution = np.exp(-3 * wa * z / (1 + z))
    de_term = OL0 * (1 + z)**de_exponent * de_evolution
    total = matter_term + de_term
    
    # Handle negative values
    total = np.maximum(total, 1e-10)
    return np.sqrt(total)

def comoving_distance_cpl(z, H0, Om0, w0, wa):
    """Comoving distance for CPL model"""
    if z <= 0:
        return 0.0
    
    c = 299792.458  # km/s
    D_H = c / H0  # Hubble distance
    
    try:
        integral, _ = quad(
            lambda zp: 1.0 / hubble_normalized_cpl(zp, H0, Om0, w0, wa),
            0, z, epsabs=1e-8, epsrel=1e-8, limit=50
        )
        return D_H * integral
    except:
        return 1e6  # Return large value for failed integration

def luminosity_distance_cpl(z, H0, Om0, w0, wa):
    """Luminosity distance for CPL model"""
    if z <= 0:
        return 1e-6
    d_c = comoving_distance_cpl(z, H0, Om0, w0, wa)
    return (1.0 + z) * d_c

def distance_modulus_cpl(z, H0, Om0, w0, wa):
    """Distance modulus for CPL model"""
    z = np.asarray(z)
    scalar_input = z.ndim == 0
    z = np.atleast_1d(z)
    
    mu = np.zeros_like(z, dtype=float)
    for i, zi in enumerate(z):
        dL = luminosity_distance_cpl(zi, H0, Om0, w0, wa)
        if dL <= 0 or not np.isfinite(dL):
            mu[i] = 50.0  # Large distance modulus for failed cases
        else:
            mu[i] = 5 * np.log10(dL) + 25
    
    return mu.item() if scalar_input else mu

# --- 2. Define the Prior Distribution ---
# Priors for [H0, Om0, w0, wa]
prior_min = torch.tensor([60.0, 0.1, -2.5, -2.0])
prior_max = torch.tensor([80.0, 0.9, 0.5, 2.0])
prior = utils.BoxUniform(low=prior_min, high=prior_max)

# --- 3. Define the Combined Simulator ---
def combined_simulator(params):
    """
    Combined simulator that generates both H(z) and SN distance modulus data.
    Args:
        params: Tensor of shape (batch_size, 4) containing [H0, Om0, w0, wa]
    Returns:
        combined_data: Tensor with H(z) data followed by SN mu data
    """
    if params.ndim == 1:
        params = params.unsqueeze(0)
    
    batch_size = params.shape[0]
    total_observations = n_hz_observations + n_sn_observations
    combined_data = torch.zeros(batch_size, total_observations)
    
    # Convert to numpy for calculations
    params_np = params.numpy()
    
    for i in range(batch_size):
        H0, Om0, w0, wa = params_np[i]
        
        # 1. Generate H(z) data
        Hz_true = hubble_cpl(z_obs, H0, Om0, w0, wa)
        noise_hz = np.random.normal(0, error_obs)
        Hz_observed = Hz_true + noise_hz
        
        # 2. Generate SN distance modulus data
        mu_true = distance_modulus_cpl(z_sn, H0, Om0, w0, wa)
        noise_sn = np.random.normal(0, mu_err_sn)
        mu_observed = mu_true + noise_sn
        
        # 3. Combine data
        combined_obs = np.concatenate([Hz_observed, mu_observed])
        combined_data[i] = torch.tensor(combined_obs)
    
    return combined_data

#########################################
#         Generate Training Data        #
#########################################
num_simulations = 2000  # Increased for joint analysis

# Sample parameters from prior
theta = prior.sample((num_simulations,))

# Run simulations
simulation_start = time.time()
print("Running combined H(z) + SN simulations...")
x = combined_simulator(theta)
simulation_end = time.time()
print(f"Combined simulations complete. Data shape: {x.shape}")
print(f"Simulation time: {simulation_end - simulation_start:.2f} seconds")

######################################### 
#  Set up and Train the Neural          # 
#  Posterior Estimator                  #
#########################################

# Initialize the inference procedure
inference = SNPE(prior=prior)

# Train the model
training_start = time.time()
print("Training neural posterior estimator on joint data...")
density_estimator = inference.append_simulations(theta, x).train()
training_end = time.time()
print("Training complete!")
print(f"Training time: {training_end - training_start:.2f} seconds")

##############################################
#  Build the Posterior and Perform Inference #
##############################################

# Build the posterior object
posterior = inference.build_posterior(density_estimator)

#########################################################
#  Use Real Observational Data                         #
#########################################################
print("Using real observational data (H(z) + SN)...")

# Combine observed data: H(z) followed by SN mu
x_observed_combined = torch.tensor(np.concatenate([hub_obs, mu_obs_sn]), dtype=torch.float32)

print(f"Combined observed data shape: {x_observed_combined.shape}")
print(f"H(z) observations: {n_hz_observations}")
print(f"SN observations: {n_sn_observations}")
print(f"Total observations: {len(x_observed_combined)}")

######################################################
#  Sample from the posterior given the observed data #
######################################################
sampling_start = time.time()
num_samples = 50000
samples = posterior.sample((num_samples,), x=x_observed_combined)
sampling_end = time.time()

print(f"Sampled {num_samples} points from the joint posterior.")
print(f"Sampling time: {sampling_end - sampling_start:.2f} seconds")

########################################################
#               Analyze the Results                    #
########################################################
# Convert samples to numpy for GetDist
samples_np = samples.numpy()

# Create GetDist samples
names = ['H0', 'Om', 'w0', 'wa']
labels = ['H_0', '\\Omega_m', 'w_0', 'w_a']
samples_gd = MCSamples(samples=samples_np, names=names, labels=labels)

# 1. Create GetDist triangle plot
print("Creating GetDist triangle plot...")
g = plots.get_subplot_plotter()
g.triangle_plot(samples_gd, filled=True, title_limit=1)
plt.suptitle('Joint Posterior Distribution from H(z) + SN Data', fontsize=16)
plt.show()

# 2. Posterior means and credible intervals
H0_samples = samples_np[:, 0]
Om0_samples = samples_np[:, 1]
w0_samples = samples_np[:, 2]
wa_samples = samples_np[:, 3]

H0_mean, H0_std = np.mean(H0_samples), np.std(H0_samples)
Om0_mean, Om0_std = np.mean(Om0_samples), np.std(Om0_samples)
w0_mean, w0_std = np.mean(w0_samples), np.std(w0_samples)
wa_mean, wa_std = np.mean(wa_samples), np.std(wa_samples)

print("\nJoint Posterior Results:")
print("="*50)
print("Posterior means:")
print(r'H0   = {:.1f} +/- {:.1f} km/s/Mpc'.format(H0_mean, H0_std))
print(r'Om = {:.3f} +/- {:.3f}'.format(Om0_mean, Om0_std))
print(r'w0 = {:.3f} +/- {:.3f}'.format(w0_mean, w0_std))
print(r'wa = {:.3f} +/- {:.3f}'.format(wa_mean, wa_std))

# Calculate credible intervals using GetDist
print("\nCredible Intervals from GetDist:")
for i, param_name in enumerate(['H0', 'Om', 'w0', 'wa']):
    stats = samples_gd.getMargeStats().parWithName(names[i])
    print(f"{param_name}:")
    print(f"  Mean +/- std: {stats.mean:.3f} +/- {stats.err:.3f}")
    print(f"  68% CI: [{stats.limits[0].lower:.3f}, {stats.limits[0].upper:.3f}]")
    print(f"  95% CI: [{stats.limits[1].lower:.3f}, {stats.limits[1].upper:.3f}]")

# --- Compare H(z)-only vs Joint constraints ---
print(f"\n" + "="*70)
print("DATA COMBINATION ANALYSIS")
print("="*70)
print(f"H(z) data points: {n_hz_observations}")
print(f"SN data points: {n_sn_observations}")
print(f"Total data points: {n_hz_observations + n_sn_observations}")
print(f"H(z) redshift range: {z_obs.min():.3f} - {z_obs.max():.3f}")
print(f"SN redshift range: {z_sn.min():.3f} - {z_sn.max():.3f}")
print(f"Combined redshift range: {min(z_obs.min(), z_sn.min()):.3f} - {max(z_obs.max(), z_sn.max()):.3f}")

# --- Data Quality Analysis ---
print(f"\n" + "="*70)
print("OBSERVATIONAL DATA SUMMARY")
print("="*70)
print(f"H(z) measurements:")
print(f"  Mean error: {error_obs.mean():.1f} km/s/Mpc")
print(f"  Mean relative error: {(error_obs/hub_obs).mean()*100:.1f}%")
print(f"SN measurements:")
print(f"  Mean error: {mu_err_sn.mean():.3f} mag")
print(f"  Mean relative error: {(mu_err_sn/mu_obs_sn).mean()*100:.1f}%")

# Calculate and print total execution time
end_time = time.time()
total_time = end_time - start_time

print(f"\n" + "="*70)
print("TIMING SUMMARY")
print("="*70)
print(f"Simulation time:  {simulation_end - simulation_start:8.2f} seconds")
print(f"Training time:    {training_end - training_start:8.2f} seconds")
print(f"Sampling time:    {sampling_end - sampling_start:8.2f} seconds")
print(f"Total time:       {total_time:8.2f} seconds ({total_time/60:.1f} minutes)")
print("="*70)