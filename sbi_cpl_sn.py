import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sbi
from sbi import analysis, utils
from sbi.inference import SNPE
from sbi.utils.user_input_checks import check_prior
from scipy.integrate import quad
import scipy.linalg as la
import pandas as pd
from getdist import plots, MCSamples

torch.manual_seed(42)
np.random.seed(42)

start_time = time.time()

# --- 1. Define the Cosmological Model (CPL) ---
def hubble_cpl(z, H0, Om0, w0, wa):
    """Hubble parameter for flat universe with CPL dark energy."""
    matter_term = Om0 * (1 + z)**3
    Ode0 = 1 - Om0 
    de_term = Ode0 * (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z))
    return H0 * np.sqrt(matter_term + de_term)

def angular_distance(z, H0, Om0, w0, wa):
    """Calculate angular diameter distance for flat universe."""
    c = 299792.458  # km/s
    
    def integrand(zp):
        return 1.0 / hubble_cpl(zp, H0, Om0, w0, wa)
    
    if np.isscalar(z):
        integral, _ = quad(integrand, 0, z)
        return c * integral / (1 + z)
    else:
        da_values = np.zeros_like(z)
        for i, zi in enumerate(z):
            integral, _ = quad(integrand, 0, zi)
            da_values[i] = c * integral / (1 + zi)
        return da_values

def distance_modulus_cosmological(z_cmb, z_hel, H0, Om0, w0, wa):
    """Calculate distance modulus for cosmological supernovae following Pantheon+ prescription."""
    da = angular_distance(z_cmb, H0, Om0, w0, wa)
    # Following the official code: mu = 5 * log10((1+z_cmb)*(1+z_hel)*d_A) + 25
    mu = 5.0 * np.log10((1 + z_cmb) * (1 + z_hel) * da) + 25.0
    return mu

# --- 2. Load Supernova Data Following Official Structure ---
def load_pantheon_data(data_filename, cov_filename):
    """
    Load Pantheon+SH0ES data following the official likelihood structure.
    """
    print(f"Loading Pantheon+SH0ES data from {data_filename}")
    
    # Load the full dataset
    data = pd.read_csv(data_filename, sep='\s+')
    origlen = len(data)
    print(f"Original dataset size: {origlen}")
    
    # Apply selection criteria following official code
    z_min = 0.01  # Minimum redshift cut
    
    # Create mask for data selection (z > z_min OR is calibrator)
    mask = ((data['zHD'] > z_min) | (data['IS_CALIBRATOR'] > 0)).values
    
    # Apply mask to data
    z_cmb = data['zHD'][mask].values
    z_hel = data['zHEL'][mask].values
    m_b_corr = data['m_b_corr'][mask].values
    is_calibrator = data['IS_CALIBRATOR'][mask].values
    ceph_dist = data['CEPH_DIST'][mask].values
    
    N = len(z_cmb)
    print(f"After selection cuts: {N} supernovae")
    print(f"Calibrators: {sum(is_calibrator)} | Cosmological: {sum(is_calibrator == 0)}")
    
    # Load and process covariance matrix following official approach
    print(f"Loading covariance matrix from {cov_filename}")
    
    # Read the covariance matrix - it's stored line by line following the official code pattern
    with open(cov_filename, 'r') as f:
        full_cov = np.zeros((origlen, origlen))
        
        # Read matrix line by line as in the official code
        for i in range(origlen):
            for j in range(origlen):
                val = float(f.readline().strip())
                full_cov[i, j] = val
    
    print(f"Loaded full covariance matrix: {full_cov.shape}")
    
    # Apply the same mask to covariance matrix
    masked_cov = full_cov[np.ix_(mask, mask)]
    
    print(f"Covariance matrix shape: {masked_cov.shape}")
    
    # Store processed data
    sn_data = {
        'z_cmb': z_cmb,
        'z_hel': z_hel,
        'm_b_corr': m_b_corr,
        'is_calibrator': is_calibrator,
        'ceph_dist': ceph_dist,
        'cov': masked_cov,
        'N': N
    }
    
    return sn_data

# Load the data
sn_data = load_pantheon_data(
    '/home/alfonsozapata/Documents/SimpleMC2/SimpleMC/simplemc/data/Pantheon+SH0ES.dat',
    '/home/alfonsozapata/Documents/SimpleMC2/SimpleMC/simplemc/data/Pantheon+SH0ES_STAT+SYS.cov'
)

# Extract data arrays
z_cmb = sn_data['z_cmb']
z_hel = sn_data['z_hel']
m_b_corr = sn_data['m_b_corr']
is_calibrator = sn_data['is_calibrator']
ceph_dist = sn_data['ceph_dist']
cov_matrix = sn_data['cov']
n_observations = sn_data['N']

print(f"\nData Summary:")
print(f"Total supernovae: {n_observations}")
print(f"Redshift range: {z_cmb.min():.3f} to {z_cmb.max():.3f}")
print(f"Magnitude range: {m_b_corr.min():.2f} to {m_b_corr.max():.2f}")

# --- 3. Define Prior Distribution ---
# Now including the nuisance parameter M
# Priors for [H0, Om0, w0, wa, M]
prior_min = torch.tensor([65.0, 0.1, -1.5, -1.5, -20.0])
prior_max = torch.tensor([75.0, 0.5, 1.5, 1.5, -18.0])
prior = utils.BoxUniform(low=prior_min, high=prior_max)

print(f"\nPrior ranges:")
print(f"H0: [{prior_min[0]:.1f}, {prior_max[0]:.1f}] km/s/Mpc")
print(f"Om0: [{prior_min[1]:.1f}, {prior_max[1]:.1f}]")
print(f"w0: [{prior_min[2]:.1f}, {prior_max[2]:.1f}]")
print(f"wa: [{prior_min[3]:.1f}, {prior_max[3]:.1f}]")
print(f"M: [{prior_min[4]:.1f}, {prior_max[4]:.1f}] mag")

# --- 4. Clean Simulator Without Cholesky ---
def simulator(params):
    """
    Clean simulator following Pantheon+SH0ES structure without Cholesky decomposition.
    Args:
        params: Tensor of shape (batch_size, 5) containing [H0, Om0, w0, wa, M]
    Returns:
        mag_data: Tensor of shape (batch_size, n_observations) with magnitude measurements
    """
    if params.ndim == 1:
        params = params.unsqueeze(0)
    
    batch_size = params.shape[0]
    mag_data = torch.zeros(batch_size, n_observations)
    params_np = params.numpy()
    
    # Extract diagonal uncertainties for noise generation (no Cholesky needed)
    cov_diagonal = np.diag(cov_matrix)
    noise_std = np.sqrt(np.maximum(cov_diagonal, 1e-10))  # Ensure positive
    
    for i in range(batch_size):
        H0, Om0, w0, wa, M = params_np[i]
        
        # Calculate theoretical distance moduli following official prescription
        mu_theory = np.zeros(n_observations)
        
        for j in range(n_observations):
            if is_calibrator[j] == 1:
                # For calibrators, use Cepheid distances
                mu_theory[j] = ceph_dist[j]
            else:
                # For cosmological SNe, calculate distance modulus
                mu_theory[j] = distance_modulus_cosmological(
                    z_cmb[j], z_hel[j], H0, Om0, w0, wa
                )
        
        # Calculate theoretical magnitudes: M + mu_theory
        theoretical_magnitudes = M + mu_theory
        
        # Generate uncorrelated noise using diagonal uncertainties
        noise = np.random.normal(0, noise_std)
        
        # Generate observed magnitudes
        mag_observed = theoretical_magnitudes + noise
        
        mag_data[i] = torch.tensor(mag_observed)
    
    return mag_data

# --- 5. Generate Training Data ---
num_simulations = 1000
print(f"\nGenerating {num_simulations} training simulations...")

# Sample parameters from prior
theta = prior.sample((num_simulations,))

# Run simulations
simulation_start = time.time()
print("Running Pantheon+SH0ES simulations...")
x = simulator(theta)
simulation_end = time.time()
print(f"Simulations complete. Data shape: {x.shape}")
print(f"Simulation time: {simulation_end - simulation_start:.2f} seconds")

# --- 6. Train Neural Posterior Estimator ---
inference = SNPE(prior=prior)

training_start = time.time()
print("Training neural posterior estimator...")
density_estimator = inference.append_simulations(theta, x).train()
training_end = time.time()
print("Training complete!")
print(f"Training time: {training_end - training_start:.2f} seconds")

# --- 7. Build Posterior ---
posterior = inference.build_posterior(density_estimator)

# --- 8. Use Real Observational Data ---
print("Using real Pantheon+SH0ES observational data...")
x_observed = torch.tensor(m_b_corr, dtype=torch.float32)
print(f"Observed data shape: {x_observed.shape}")

# --- 9. Sample from Posterior ---
sampling_start = time.time()
num_samples = 5000
samples = posterior.sample((num_samples,), x=x_observed)
sampling_end = time.time()

print(f"Sampled {num_samples} points from the posterior.")
print(f"Sampling time: {sampling_end - sampling_start:.2f} seconds")

# --- 10. Analysis and Results ---
samples_np = samples.numpy()

# Create GetDist samples
names = ['H0', 'Om', 'w0', 'wa', 'M']
labels = ['H_0', '\\Omega_m', 'w_0', 'w_a', 'M']
samples_gd = MCSamples(samples=samples_np, names=names, labels=labels)

# Triangle plot
print("Creating GetDist triangle plot...")
g = plots.get_subplot_plotter()
g.triangle_plot(samples_gd, filled=True, title_limit=1)
plt.suptitle('Pantheon+SH0ES Posterior Distribution (SBI)', fontsize=16)
plt.show()

# Posterior statistics
param_names = ['H0', 'Om', 'w0', 'wa', 'M']
param_units = ['km/s/Mpc', '', '', '', 'mag']

print("\nPosterior Results:")
print("="*60)
for i, (name, unit) in enumerate(zip(param_names, param_units)):
    samples_param = samples_np[:, i]
    mean = np.mean(samples_param)
    std = np.std(samples_param)
    print(f"{name:3s} = {mean:7.3f} ± {std:6.3f} {unit}")

# GetDist credible intervals
print("\nCredible Intervals:")
print("="*60)
for i, param_name in enumerate(param_names):
    stats = samples_gd.getMargeStats().parWithName(names[i])
    print(f"{param_name}:")
    print(f"  68% CI: [{stats.limits[0].lower:.3f}, {stats.limits[0].upper:.3f}]")
    print(f"  95% CI: [{stats.limits[1].lower:.3f}, {stats.limits[1].upper:.3f}]")

# --- 11. Model Validation ---
def plot_hubble_diagram():
    """Plot Hubble diagram with best-fit model"""
    # Get best-fit parameters (posterior mean)
    best_params = samples_np.mean(axis=0)
    H0_best, Om0_best, w0_best, wa_best, M_best = best_params
    
    # Calculate theoretical distance moduli
    mu_theory = np.zeros(n_observations)
    for j in range(n_observations):
        if is_calibrator[j] == 1:
            mu_theory[j] = ceph_dist[j]
        else:
            mu_theory[j] = distance_modulus_cosmological(
                z_cmb[j], z_hel[j], H0_best, Om0_best, w0_best, wa_best
            )
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Separate calibrators and cosmological SNe
    calib_mask = is_calibrator == 1
    cosmo_mask = is_calibrator == 0
    
    # Plot data
    if np.any(calib_mask):
        plt.errorbar(z_cmb[calib_mask], m_b_corr[calib_mask], 
                    yerr=np.sqrt(np.diag(cov_matrix)[calib_mask]),
                    fmt='o', color='red', alpha=0.7, label='Cepheid Calibrators')
    
    plt.errorbar(z_cmb[cosmo_mask], m_b_corr[cosmo_mask], 
                yerr=np.sqrt(np.diag(cov_matrix)[cosmo_mask]),
                fmt='o', color='blue', alpha=0.6, label='Cosmological SNe')
    
    # Plot theoretical predictions
    theoretical_mags = M_best + mu_theory
    plt.plot(z_cmb, theoretical_mags, 'k-', linewidth=2, label='Best-fit Model')
    
    plt.xlabel('Redshift z')
    plt.ylabel('Apparent Magnitude m_b')
    plt.title('Pantheon+SH0ES Hubble Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, z_cmb.max() * 1.05)
    plt.show()

# Create Hubble diagram
plot_hubble_diagram()

# --- 12. Data Quality Summary ---
print(f"\n" + "="*70)
print("PANTHEON+SH0ES DATA SUMMARY")
print("="*70)
print(f"Total supernovae: {n_observations}")
print(f"Cepheid calibrators: {sum(is_calibrator)}")
print(f"Cosmological SNe: {sum(is_calibrator == 0)}")
print(f"Redshift range: {z_cmb.min():.4f} - {z_cmb.max():.3f}")
print(f"Magnitude range: {m_b_corr.min():.2f} - {m_b_corr.max():.2f}")

# Covariance analysis
cov_diag = np.diag(cov_matrix)
print(f"Mean magnitude uncertainty: {np.sqrt(cov_diag).mean():.3f} mag")
print(f"Uncertainty range: {np.sqrt(cov_diag).min():.3f} - {np.sqrt(cov_diag).max():.3f} mag")
print(f"Covariance condition number: {np.linalg.cond(cov_matrix):.1e}")

# Timing summary
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

print("\n" + "="*70)
print("IMPLEMENTATION NOTES")
print("="*70)
print("✓ No Cholesky decomposition - diagonal noise only")
print("✓ Proper calibrator/cosmological SN distinction")
print("✓ Nuisance parameter M included")
print("✓ Correct distance modulus calculation")
print("✓ Following official Pantheon+SH0ES prescription")
print("✓ Numerically stable implementation")
print("="*70)