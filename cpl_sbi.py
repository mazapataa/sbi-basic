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
#arr_hub = np.loadtxt('/home/alfonsozapata/SimpleMC/simplemc/data/Hz_all.dat')
arr_hub = np.loadtxt('/Users/alfonsozapata/Documents/SimpleMC/simplemc/data/Hz_all.dat') 
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

print(f"Reference parameters: H0 = {H0_true}, Om = {Om0_true}, w0 = {w0_true}, wa = {wa_true}")
print(f"Observed H(z) data shape: {x_observed.shape}")

# --- 8. Sample from the Posterior ---
# Sample from the posterior given the observed data
sampling_start = time.time()
num_samples = 50000
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
print(r'$H_0$   = {H0_mean:.1f} +/- {H0_std:.1f} km/s/Mpc')
print(r"$\Omega_m$ = {Om0_mean:.3f} +/-  {Om0_std:.3f}")
print(r' $w_0$ = {w0_mean:.3f} +/- {w0_std:.3f} ')
print(r'  $w_a$ = {wa_mean:.3f} +/- {wa_std:.3f} ')





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



