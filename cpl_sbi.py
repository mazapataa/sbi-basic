import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Import SBI components - FIXED IMPORTS
import sbi
from sbi import analysis, utils
from sbi.inference import SNPE
from sbi.utils.user_input_checks import check_prior

# For cosmology calculations
from scipy.integrate import quad

# Import GetDist for corner plots
from getdist import plots, MCSamples

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Start timing
start_time = time.time()

# --- 1. Define the Cosmological Model (CPL) ---
# The CPL parameterization: w(z) = w0 + wa * z / (1 + z)
# We'll use a simple flat universe: H(z)^2 = H0^2 [ Ω_m(1+z)^3 + Ω_DE(z) ]

# Load real observational data
arr_hub = np.loadtxt('/home/alfonsozapata/SimpleMC/simplemc/data/Hz_all.dat')
z_obs = arr_hub[:,0]
hub_obs = arr_hub[:,1]
error_obs = arr_hub[:,2]  # Real observational errors
n_observations = len(z_obs)

print(f"Loaded {n_observations} H(z) measurements")
print(f"Redshift range: {z_obs.min():.3f} to {z_obs.max():.3f}")
print(f"H(z) range: {hub_obs.min():.1f} to {hub_obs.max():.1f} km/s/Mpc")
print(f"Error range: {error_obs.min():.1f} to {error_obs.max():.1f} km/s/Mpc")

def hubble_cpl(z, H0, Om0, w0, wa):
    """
    Hubble parameter for flat universe with CPL dark energy.
    Ω_DE = 1 - Ω_m (flat universe)
    """
    matter_term = Om0 * (1 + z)**3
    Ode0 = 1 - Om0  # Flat universe constraint
    de_term = Ode0 * (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z))
    return H0 * np.sqrt(matter_term + de_term)

# --- 2. Define the Prior Distribution ---
# We assume uniform priors on H0, Om0, w0, and wa
prior_min = torch.tensor([60.0, 0.1, -2.5, -2.0])  # [H0, Om0, w0, wa]
prior_max = torch.tensor([80.0, 0.9, -0.0, 2.0])   # [H0, Om0, w0, wa]
prior = utils.BoxUniform(low=prior_min, high=prior_max)

# --- 3. Define the Simulator ---
def simulator(params):
    """
    Simulator that generates H(z) data given cosmological parameters using real observational errors.
    Args:
        params: Tensor of shape (batch_size, 4) containing [H0, Om0, w0, wa]
    Returns:
        H_data: Tensor of shape (batch_size, n_observations) with H(z) measurements + noise
    """
    # Ensure params is 2D
    if params.ndim == 1:
        params = params.unsqueeze(0)
    
    batch_size = params.shape[0]
    H_data = torch.zeros(batch_size, n_observations)
    
    # Convert to numpy for calculations
    params_np = params.numpy()
    
    for i in range(batch_size):
        H0, Om0, w0, wa = params_np[i]
        
        # Calculate true H(z) without noise
        Hz_true = hubble_cpl(z_obs, H0, Om0, w0, wa)
        
        # Add realistic observational noise using actual error bars
        noise = np.random.normal(0, error_obs)  # Use real observational errors
        Hz_observed = Hz_true + noise
        
        H_data[i] = torch.tensor(Hz_observed)
    
    return H_data

# --- 4. Generate Training Data ---
num_simulations = 1000  # Reduced for testing, increase for production

# Sample parameters from prior
theta = prior.sample((num_simulations,))

# Run simulations (this might take a while)
simulation_start = time.time()
print("Running simulations...")
x = simulator(theta)
simulation_end = time.time()
print(f"Simulations complete. Data shape: {x.shape}")
print(f"Simulation time: {simulation_end - simulation_start:.2f} seconds")

# --- 5. Set up and Train the Neural Posterior Estimator ---
# Initialize the inference procedure with default neural network
inference = SNPE(prior=prior)

# Train the model
training_start = time.time()
print("Training neural posterior estimator...")
density_estimator = inference.append_simulations(theta, x).train()
training_end = time.time()
print("Training complete!")
print(f"Training time: {training_end - training_start:.2f} seconds")

# --- 6. Build the Posterior and Perform Inference ---
# Build the posterior object that can be sampled and evaluated
posterior = inference.build_posterior(density_estimator)

# --- 7. Use Real Observational Data or Generate Mock Data ---
# Option 1: Use the real observational data
print("Using real observational data...")
x_observed = torch.tensor(hub_obs, dtype=torch.float32)

# Option 2: Or generate mock data based on true parameters
H0_true, Om0_true, w0_true, wa_true = 71.0, 0.3, -1.0, 0.0  # ΛCDM values
true_params = torch.tensor([H0_true, Om0_true, w0_true, wa_true])

# Uncomment the following lines if you want to use mock data instead of real data:
# print("Generating 'observed' data...")
# x_observed = simulator(true_params.unsqueeze(0))[0]  # Remove batch dimension

print(f"Reference parameters: H₀ = {H0_true}, Ωₘ = {Om0_true}, w₀ = {w0_true}, wₐ = {wa_true}")
print(f"Observed H(z) data shape: {x_observed.shape}")

# --- 8. Sample from the Posterior ---
# Sample from the posterior given the observed data
sampling_start = time.time()
num_samples = 5000
samples = posterior.sample((num_samples,), x=x_observed)
sampling_end = time.time()

print(f"Sampled {num_samples} points from the posterior.")
print(f"Sampling time: {sampling_end - sampling_start:.2f} seconds")

# --- 9. Analyze and Plot the Results ---
# Convert to numpy for plotting
samples_np = samples.numpy()
true_params_np = torch.tensor([H0_true, Om0_true, w0_true, wa_true]).numpy()

# Create GetDist samples
names = ['H0', 'Om', 'w0', 'wa']
labels = ['H_0', '\\Omega_m', 'w_0', 'w_a']
samples_gd = MCSamples(samples=samples_np, names=names, labels=labels)

# 1. Create GetDist triangle plot
print("Creating GetDist triangle plot...")
g = plots.get_subplot_plotter()
g.triangle_plot(samples_gd, filled=True, title_limit=1)
plt.suptitle('Posterior Distribution from H(z) Data', fontsize=16)
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

print(f"Posterior means:")
print(f"  H₀ = {H0_mean:.1f} ± {H0_std:.1f} km/s/Mpc")
print(f"  Ωₘ = {Om0_mean:.3f} ± {Om0_std:.3f}")
print(f"  w₀ = {w0_mean:.3f} ± {w0_std:.3f}")
print(f"  wₐ = {wa_mean:.3f} ± {wa_std:.3f}")

# 3. Create individual plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot the Hubble data with constraints
ax1 = axes[0, 0]
# Plot the true underlying relation (using ΛCDM as reference)
z_plot = np.linspace(z_obs.min(), z_obs.max(), 100)
Hz_true_plot = hubble_cpl(z_plot, H0_true, Om0_true, w0_true, wa_true)
ax1.plot(z_plot, Hz_true_plot, 'r-', label='ΛCDM reference', lw=2)

# Plot the observational data points with REAL error bars
ax1.errorbar(z_obs, x_observed, yerr=error_obs, fmt='o', 
            capsize=5, label='Observational data', color='blue', alpha=0.7)

ax1.set_xlabel('Redshift z')
ax1.set_ylabel('H(z) [km/s/Mpc]')
ax1.set_title('Hubble Parameter Data')
ax1.legend()
ax1.grid(True)

# Plot the equation of state evolution
ax2 = axes[0, 1]
# True evolution (ΛCDM)
w_true_plot = w0_true + wa_true * z_plot / (1 + z_plot)
ax2.plot(z_plot, w_true_plot, 'r-', label='ΛCDM (w = -1)', lw=2)

# Posterior samples of w(z)
for i in range(100):  # Plot a subset of samples
    H0_sample, Om0_sample, w0_sample, wa_sample = samples_np[i]
    w_sample = w0_sample + wa_sample * z_plot / (1 + z_plot)
    ax2.plot(z_plot, w_sample, 'gray', alpha=0.1)

ax2.axhline(y=-1, color='k', linestyle='--', alpha=0.7, label='w = -1')
ax2.set_xlabel('Redshift z')
ax2.set_ylabel('w(z)')
ax2.set_title('Dark Energy Equation of State')
ax2.set_ylim(-2.5, 0.5)
ax2.legend()
ax2.grid(True)

# Posterior predictive check
ax3 = axes[1, 0]
# Plot the observed data again with real errors
ax3.errorbar(z_obs, x_observed, yerr=error_obs, fmt='o', 
            capsize=5, label='Data', color='blue', alpha=0.7)

# Plot predictions from posterior samples
for i in range(50):  # Plot a subset of samples
    H0_sample, Om0_sample, w0_sample, wa_sample = samples_np[i]
    Hz_sample = hubble_cpl(z_obs, H0_sample, Om0_sample, w0_sample, wa_sample)
    ax3.plot(z_obs, Hz_sample, 'gray', alpha=0.1, lw=1)

# Plot the mean prediction
mean_Hz = np.mean([hubble_cpl(z_obs, H0, Om0, w0, wa)
                  for H0, Om0, w0, wa in samples_np[::100]], axis=0)
ax3.plot(z_obs, mean_Hz, 'orange', lw=3, label='Posterior mean')

ax3.set_xlabel('Redshift z')
ax3.set_ylabel('H(z) [km/s/Mpc]')
ax3.set_title('Posterior Predictive Check')
ax3.legend()
ax3.grid(True)

# Plot H₀ and Ωₘ posterior distributions
ax4 = axes[1, 1]
# H₀ distribution
ax4_hist1 = ax4.twinx()
n1, bins1, patches1 = ax4.hist(H0_samples, bins=30, density=True, alpha=0.7, 
                              color='skyblue', edgecolor='black', label='H₀')
ax4.axvline(H0_mean, color='blue', linestyle='--', label=f'H₀ mean = {H0_mean:.1f}')
ax4.axvline(H0_true, color='blue', linestyle=':', label=f'H₀ ref = {H0_true:.1f}')

# Ωₘ distribution
n2, bins2, patches2 = ax4_hist1.hist(Om0_samples, bins=30, density=True, alpha=0.7, 
                                    color='lightcoral', edgecolor='black', label='Ωₘ')
ax4_hist1.axvline(Om0_mean, color='red', linestyle='--', label=f'Ωₘ mean = {Om0_mean:.3f}')
ax4_hist1.axvline(Om0_true, color='red', linestyle=':', label=f'Ωₘ ref = {Om0_true:.3f}')

ax4.set_xlabel('Parameter Value')
ax4.set_ylabel('H₀ Probability Density', color='blue')
ax4_hist1.set_ylabel('Ωₘ Probability Density', color='red')
ax4.set_title('H₀ and Ωₘ Posterior Distributions')
ax4.grid(True, alpha=0.3)

# Combine legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_hist1.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.show()

# --- 10. Calculate Credible Intervals ---
# Calculate 68% and 95% credible intervals using GetDist
print("\nCredible Intervals from GetDist:")
for i, param_name in enumerate(['H₀', 'Ωₘ', 'w₀', 'wₐ']):
    stats = samples_gd.getMargeStats().parWithName(names[i])
    print(f"{param_name}:")
    print(f"  Mean ± std: {stats.mean:.3f} ± {stats.err:.3f}")
    print(f"  68% CI: [{stats.limits[0].lower:.3f}, {stats.limits[0].upper:.3f}]")
    print(f"  95% CI: [{stats.limits[1].lower:.3f}, {stats.limits[1].upper:.3f}]")

# --- 11. Data Quality Analysis ---
print(f"\n" + "="*70)
print("OBSERVATIONAL DATA SUMMARY")
print("="*70)
print(f"Number of data points: {n_observations}")
print(f"Redshift range: {z_obs.min():.3f} - {z_obs.max():.3f}")
print(f"H(z) range: {hub_obs.min():.1f} - {hub_obs.max():.1f} km/s/Mpc")
print(f"Mean error: {error_obs.mean():.1f} km/s/Mpc")
print(f"Error range: {error_obs.min():.1f} - {error_obs.max():.1f} km/s/Mpc")
print(f"Mean relative error: {(error_obs/hub_obs).mean()*100:.1f}%")

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

# --- 12. Additional Analysis: Check for ΛCDM consistency ---
# Count how many samples are consistent with ΛCDM (w₀ = -1, wₐ = 0)
lcdm_mask = (np.abs(w0_samples + 1) < 0.1) & (np.abs(wa_samples) < 0.1)
lcdm_fraction = np.mean(lcdm_mask) * 100
print(f"\nFraction of posterior consistent with ΛCDM (|w₀+1| < 0.1, |wₐ| < 0.1): {lcdm_fraction:.1f}%")

# Check for Ωₘ constraints relative to Planck
planck_Om0 = 0.315
planck_mask = (np.abs(Om0_samples - planck_Om0) < 0.05)
planck_fraction = np.mean(planck_mask) * 100
print(f"Fraction of posterior consistent with Planck Ωₘ (0.315 ± 0.05): {planck_fraction:.1f}%")

# Check for H₀ constraints relative to common values
h0_shoES = 73.0  # SH0ES value
h0_planck = 67.4  # Planck value
shoes_mask = (np.abs(H0_samples - h0_shoES) < 2.0)
planck_h0_mask = (np.abs(H0_samples - h0_planck) < 2.0)
shoes_fraction = np.mean(shoes_mask) * 100
planck_h0_fraction = np.mean(planck_h0_mask) * 100
print(f"Fraction of posterior consistent with SH0ES H₀ (73.0 ± 2.0): {shoes_fraction:.1f}%")
print(f"Fraction of posterior consistent with Planck H₀ (67.4 ± 2.0): {planck_h0_fraction:.1f}%")

# --- 13. Parameter Correlations ---
print(f"\nParameter correlations:")
correlation_matrix = np.corrcoef(samples_np.T)
param_names = ['H₀', 'Ωₘ', 'w₀', 'wₐ']
for i in range(4):
    for j in range(i+1, 4):
        print(f"  {param_names[i]} - {param_names[j]}: {correlation_matrix[i,j]:.3f}")

# --- 14. Additional GetDist Plots ---
# Create a separate plot for 1D distributions
g = plots.get_single_plotter()
g.plot_1d(samples_gd, 'H0', normalized=True)
plt.title('H₀ Posterior Distribution')
plt.show()

g.plot_1d(samples_gd, 'Om', normalized=True)
plt.title('Ωₘ Posterior Distribution')
plt.show()

# Create a 2D plot for H0 vs Om
g = plots.get_single_plotter()
g.plot_2d(samples_gd, 'H0', 'Om', filled=True)
plt.title('H₀ vs Ωₘ Joint Posterior')
plt.show()