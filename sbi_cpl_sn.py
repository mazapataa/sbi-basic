import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sbi
from sbi import analysis, utils
from sbi.inference import SNPE
from sbi.utils.user_input_checks import check_prior
from scipy.integrate import quad
from scipy.interpolate import interp1d
import scipy.linalg as la
import pandas as pd
from getdist import plots, MCSamples

torch.manual_seed(42)
np.random.seed(42)

start_time = time.time()

# --- 1. Define the Cosmological Model (CPL) ---
# The CPL parameterization: w(z) = w0 + wa * z / (1 + z)
# We'll use a flat universe: H(z)^2 = H0^2 [ Ω_m(1+z)^3 + Ω_DE(z) ]

def hubble_cpl(z, H0, Om0, w0, wa):
    """
    Hubble parameter for flat universe with CPL dark energy.
    """
    matter_term = Om0 * (1 + z)**3
    Ode0 = 1 - Om0 
    de_term = Ode0 * (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z))
    return H0 * np.sqrt(matter_term + de_term)

def distance_modulus_cpl(z, H0, Om0, w0, wa):
    """
    Calculate distance modulus for CPL cosmology.
    """
    c = 299792.458  # km/s
    
    # Luminosity distance calculation
    def integrand(zp):
        return 1.0 / hubble_cpl(zp, H0, Om0, w0, wa)
    
    # Vectorized integration for multiple redshifts
    if np.isscalar(z):
        z = np.array([z])
        scalar_input = True
    else:
        scalar_input = False
        
    dl_values = np.zeros_like(z)
    for i, zi in enumerate(z):
        integral, _ = quad(integrand, 0, zi)
        dl = c * (1 + zi) * integral
        dl_values[i] = dl
    
    # Distance modulus: mu = 5 * log10(dL/Mpc) + 25
    mu = 5.0 * np.log10(dl_values) + 25.0
    
    if scalar_input:
        return mu[0]
    return mu

# --- 2. Load Supernova Data ---
def load_supernova_data(values_filename, cov_filename):
    """
    Load supernova data and covariance matrix similar to PantheonPlusSNLikelihood
    """
    print(f"Loading supernova data from {values_filename}")
    
    # Load data
    data = pd.read_csv(values_filename, sep='\s+')
    
    # Apply quality cuts similar to original code
    origlen = len(data)
    ww = (data['zHD'] > 0.01)  # Remove very low-z objects
    
    zcmb = data['zHD'][ww].values
    zhelio = data['zHEL'][ww].values
    mag = data['m_b_corr'][ww].values
    N = len(mag)
    
    print(f"Applied z > 0.01 cut: {origlen} -> {N} supernovae")
    
    # Load covariance matrix
    print(f"Loading covariance matrix from {cov_filename}")
    with open(cov_filename, 'r') as f:
        n = N
        C = np.zeros((n, n))
        ii = -1
        
        for i in range(origlen):
            jj = -1
            if ww[i]:
                ii += 1
            for j in range(origlen):
                if ww[j]:
                    jj += 1
                val = float(f.readline())
                if ww[i] and ww[j]:
                    C[ii, jj] = val
                    
    # Add systematic floor (similar to original code)
    C += 3**2  # Add to diagonal
    print(f"Loaded {N}x{N} covariance matrix")
    
    # Store data in a structure
    sn_data = {
        'z': zcmb,
        'mag': mag,
        'cov': C,
        'icov': la.inv(C),
        'N': N
    }
    
    return sn_data

# Load supernova data
sn_data = load_supernova_data(
    '/Users/alfonsozapata/Documents/SimpleMC/simplemc/data/Pantheon+SH0ES.dat',
    '/Users/alfonsozapata/Documents/SimpleMC/simplemc/data/Pantheon+SH0ES_STAT+SYS.cov'
)

z_obs = sn_data['z']
mag_obs = sn_data['mag']
cov_matrix = sn_data['cov']
icov_matrix = sn_data['icov']
n_observations = sn_data['N']

print(f"Loaded {n_observations} supernova measurements")
print(f"Redshift range: {z_obs.min():.3f} to {z_obs.max():.3f}")
print(f"Magnitude range: {mag_obs.min():.2f} to {mag_obs.max():.2f}")

# --- 3. Define the Prior Distribution ---
# Priors for [H0, Om0, w0, wa]
prior_min = torch.tensor([60.0, 0.1, -2.5, -2.0])
prior_max = torch.tensor([80.0, 0.9, -0.0, 2.0])
prior = utils.BoxUniform(low=prior_min, high=prior_max)

# --- 4. Define the Simulator ---
def simulator(params):
    """
    Simulator that generates supernova magnitude data given cosmological parameters.
    Args:
        params: Tensor of shape (batch_size, 4) containing [H0, Om0, w0, wa]
    Returns:
        mag_data: Tensor of shape (batch_size, n_observations) with magnitude measurements
    """
    if params.ndim == 1:
        params = params.unsqueeze(0)
    
    batch_size = params.shape[0]
    mag_data = torch.zeros(batch_size, n_observations)
    
    # Convert to numpy for calculations
    params_np = params.numpy()
    
    for i in range(batch_size):
        H0, Om0, w0, wa = params_np[i]
        
        mu_theory = distance_modulus_cpl(z_obs, H0, Om0, w0, wa)

            # If Cholesky fails, use diagonal approximation
        noise = np.random.normal(0, np.sqrt(np.diag(cov_matrix)))
        
        # Generate observed magnitudes with noise
        mag_observed = mu_theory + noise
        
        mag_data[i] = torch.tensor(mag_observed)
    
    return mag_data

#########################################
#         Generate Training Data        #
#########################################
num_simulations = 2000  # Increased for supernova analysis

# Sample parameters from prior
theta = prior.sample((num_simulations,))

# Run simulations
simulation_start = time.time()
print("Running supernova simulations...")
x = simulator(theta)
simulation_end = time.time()
print(f"Simulations complete. Data shape: {x.shape}")
print(f"Simulation time: {simulation_end - simulation_start:.2f} seconds")

######################################### 
#  Set up and Train the Neural          # 
#  Posterior Estimator                  #
#########################################

# Initialize the inference procedure
inference = SNPE(prior=prior)

# Train the model
training_start = time.time()
print("Training neural posterior estimator...")
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
print("Using supernova observational data...")
x_observed = torch.tensor(mag_obs, dtype=torch.float32)
print(f"Observed supernova data shape: {x_observed.shape}")

######################################################
#  Sample from the posterior given the observed data #
######################################################
sampling_start = time.time()
num_samples = 50000
samples = posterior.sample((num_samples,), x=x_observed)
sampling_end = time.time()

print(f"Sampled {num_samples} points from the posterior.")
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
plt.suptitle('Posterior Distribution from Supernova Data', fontsize=16)
plt.show()

# 2. Posterior statistics
H0_samples = samples_np[:, 0]
Om0_samples = samples_np[:, 1]
w0_samples = samples_np[:, 2]
wa_samples = samples_np[:, 3]

H0_mean, H0_std = np.mean(H0_samples), np.std(H0_samples)
Om0_mean, Om0_std = np.mean(Om0_samples), np.std(Om0_samples)
w0_mean, w0_std = np.mean(w0_samples), np.std(w0_samples)
wa_mean, wa_std = np.mean(wa_samples), np.std(wa_samples)

print("Posterior means:")
print(f'H0 = {H0_mean:.1f} +/- {H0_std:.1f} km/s/Mpc')
print(f'Om = {Om0_mean:.3f} +/- {Om0_std:.3f}')
print(f'w0 = {w0_mean:.3f} +/- {w0_std:.3f}')
print(f'wa = {wa_mean:.3f} +/- {wa_std:.3f}')

# Calculate credible intervals
print("\nCredible Intervals from GetDist:")
for i, param_name in enumerate(['H0', 'Om', 'w0', 'wa']):
    stats = samples_gd.getMargeStats().parWithName(names[i])
    print(f"{param_name}:")
    print(f"  Mean +/- std: {stats.mean:.3f} +/- {stats.err:.3f}")
    print(f"  68% CI: [{stats.limits[0].lower:.3f}, {stats.limits[0].upper:.3f}]")
    print(f"  95% CI: [{stats.limits[1].lower:.3f}, {stats.limits[1].upper:.3f}]")

# --- Data Quality Analysis ---
print(f"\n" + "="*70)
print("SUPERNOVA DATA SUMMARY")
print("="*70)
print(f"Number of supernovae: {n_observations}")
print(f"Redshift range: {z_obs.min():.3f} - {z_obs.max():.3f}")
print(f"Magnitude range: {mag_obs.min():.2f} - {mag_obs.max():.2f}")

# Analyze covariance structure
cov_diag = np.diag(cov_matrix)
print(f"Mean magnitude uncertainty: {np.sqrt(cov_diag).mean():.3f} mag")
print(f"Uncertainty range: {np.sqrt(cov_diag).min():.3f} - {np.sqrt(cov_diag).max():.3f} mag")

# Calculate condition number of covariance matrix
cond_num = np.linalg.cond(cov_matrix)
print(f"Covariance matrix condition number: {cond_num:.1e}")

# Calculate total execution time
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

# --- Optional: Model comparison plot ---
def plot_model_comparison():
    """
    Plot observed vs predicted magnitudes for best-fit parameters
    """
    # Get best-fit parameters (posterior mean)
    best_params = samples_np.mean(axis=0)
    H0_best, Om0_best, w0_best, wa_best = best_params
    
    # Calculate theoretical predictions
    mu_theory = distance_modulus_cpl(z_obs, H0_best, Om0_best, w0_best, wa_best)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(z_obs, mag_obs, yerr=np.sqrt(np.diag(cov_matrix)), 
                fmt='o', alpha=0.6, label='Observed')
    plt.plot(z_obs, mu_theory, 'r-', label='Best-fit model')
    plt.xlabel('Redshift z')
    plt.ylabel('Distance Modulus μ')
    plt.legend()
    plt.title('Supernova Hubble Diagram')
    plt.grid(True, alpha=0.3)
    plt.show()

# Uncomment to create model comparison plot
# plot_model_comparison()