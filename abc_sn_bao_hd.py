import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Tuple, Optional, Callable, Union, Dict, List
import warnings
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CosmologyCalculator:
    """
    A class for cosmological calculations with CPL parameterization for dark energy.
    
    The CPL (Chevallier-Polarski-Linder) parameterization: w(z) = w0 + wa * z/(1+z)
    """
    
    def __init__(self, Omega_m0: float = 0.3, Omega_L0: float = 0.7, h: float = 0.71):
     
        """
        Initialize cosmological parameters.
        """
        self.Omega_m0 = Omega_m0
        self.Omega_L0 = Omega_L0
        self.h = h
        self.H0 = h * 100  # km/s/Mpc
        self.D_H = 2997.98 / h  # Hubble distance in Mpc
        
        # Cache for interpolators
        self._dL_interp = None
        self._zmax_interp = None
        
        # Validate cosmological parameters
        if not np.isclose(Omega_m0 + Omega_L0, 1.0, rtol=1e-3):
            warnings.warn(f"Omega_m0 + Omega_L0 = {Omega_m0 + Omega_L0:.3f} ≠ 1.0. "
                         "Assuming flat universe.")
    
    def hubble_normalized_cpl(self, z: Union[float, np.ndarray], w0: float, wa: float) -> Union[float, np.ndarray]:
        """
        Normalized Hubble parameter H(z)/H0 for CPL dark energy model.
        """
        z = np.asarray(z)
        
        # Matter term
        matter_term = self.Omega_m0 * (1 + z)**3
        
        # Dark energy term with CPL parameterization
        de_exponent = 3 * (1 + w0 + wa)
        de_evolution = np.exp(-3 * wa * z / (1 + z))
        de_term = self.Omega_L0 * (1 + z)**de_exponent * de_evolution
        
        # Check for numerical issues
        total = matter_term + de_term
        if np.any(total <= 0):
            raise ValueError("Negative energy density encountered. Check parameters.")
            
        return np.sqrt(total)
    
    def inverse_hubble_normalized_cpl(self, z: float, w0: float, wa: float) -> float:
        """Inverse of normalized Hubble parameter for integration."""
        try:
            return 1.0 / self.hubble_normalized_cpl(z, w0, wa)
        except (ValueError, ZeroDivisionError):
            return np.inf
    
    def comoving_distance_cpl(self, z: float, w0: float, wa: float) -> float:
        """
        Comoving distance in Mpc for CPL model.
        """
        if z <= 0:
            return 0.0
            
        try:
            integral, error = quad(
                lambda zp: self.inverse_hubble_normalized_cpl(zp, w0, wa),
                0, z, 
                epsabs=1e-6,  # Reduced tolerance for speed
                epsrel=1e-6,  # Reduced tolerance for speed
                limit=100     # Reduced limit for speed
            )
            
            # Check integration error
            if error > 1e-4 * integral:  # More lenient error check
                warnings.warn(f"Large integration error at z={z}: {error}")
                
            return self.D_H * integral
            
        except Exception as e:
            logger.error(f"Integration failed at z={z}: {e}")
            return np.inf
    
    def angular_diameter_distance_cpl(self, z: float, w0: float, wa: float) -> float:
        """Angular diameter distance in Mpc."""
        if z <= 0:
            return 0.0
        return self.comoving_distance_cpl(z, w0, wa) / (1.0 + z)
    
    def luminosity_distance_cpl(self, z: float, w0: float, wa: float) -> float:
        """Luminosity distance in Mpc."""
        if z <= 0:
            return 0.0
        return (1.0 + z) * self.comoving_distance_cpl(z, w0, wa)
    
    def build_luminosity_distance_interpolator(self, w0: float, wa: float, 
                                             zmax: float = 2.0, npoints: int = 100):
        """
        Build interpolator for faster luminosity distance calculations.
        """
        zgrid = np.linspace(1e-4, zmax, npoints)
        dLgrid = np.array([self.luminosity_distance_cpl(z, w0, wa) for z in zgrid])
        
        # Check for invalid values
        if np.any(~np.isfinite(dLgrid)):
            raise ValueError("Invalid values in luminosity distance grid")
            
        self._dL_interp = interp1d(zgrid, dLgrid, kind='cubic', 
                                  bounds_error=False, fill_value='extrapolate')
        self._zmax_interp = zmax
        
        logger.info(f"Built luminosity distance interpolator for z ∈ [0, {zmax}]")
    
    def luminosity_distance_fast(self, z: Union[float, np.ndarray], 
                               w0: float, wa: float) -> Union[float, np.ndarray]:
        """
        Fast luminosity distance using interpolation when available.
        """
        z = np.asarray(z)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)
        
        # Use interpolator if available and within range
        if (self._dL_interp is not None and 
            self._zmax_interp is not None and 
            np.all(z <= self._zmax_interp)):
            result = self._dL_interp(z)
        else:
            result = np.array([self.luminosity_distance_cpl(zi, w0, wa) for zi in z])
        
        return result.item() if scalar_input else result
    
    def distance_modulus(self, z: Union[float, np.ndarray], 
                        w0: float, wa: float) -> Union[float, np.ndarray]:
        """
        Distance modulus μ = 5 log₁₀(dL/Mpc) + 25.
        """
        try:
            dL = self.luminosity_distance_fast(z, w0, wa)
            
            # Check for invalid distances
            if np.any(dL <= 0) or np.any(~np.isfinite(dL)):
                return np.full_like(z, 1e6, dtype=float)  # Large penalty
                
            return 5 * np.log10(dL) + 25
            
        except Exception as e:
            logger.error(f"Distance modulus calculation failed: {e}")
            return np.full_like(z, 1e6, dtype=float)


class DataLoader:
    """Class for loading and managing observational data."""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self._sn_data = None
        self._sn_cov = None
        self._hubble_data = None
    
    def load_pantheon_data(self, filename: str = "binned_pantheon.txt") -> pd.DataFrame:
        """Load binned Pantheon supernova data."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        try:
            data = pd.read_csv(filepath, sep=r'\s+')
            
            # Find the redshift column (more robust)
            z_col = None
            for col in data.columns:
                if col.lower() in ['zcmb', 'z', 'redshift', 'zhel']:
                    z_col = col
                    break
                    
            if z_col is None:
                raise ValueError("Could not find redshift column in SN data")
                
            # Find the magnitude column
            mag_col = None
            for col in data.columns:
                if col.lower() in ['mb', 'm_b', 'mag', 'magnitude']:
                    mag_col = col
                    break
                    
            if mag_col is None:
                raise ValueError("Could not find magnitude column in SN data")
                
            # Select only the needed columns
            data = data[[z_col, mag_col]].copy()
            data.columns = ['zCMB', 'mb']
            
            self._sn_data = data
            logger.info(f"Loaded {len(data)} SN data points from {filename}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load Pantheon data: {e}")
            raise
    
    def load_covariance_matrix(self, filename: str = "binned_cov_pantheon.txt") -> np.ndarray:
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Covariance file not found: {filepath}")

        try:
            # Read all lines and try to parse as floats
            with open(filepath) as f:
                lines = f.readlines()

            # Try to parse first line as integer (original format)
            try:
                n = int(lines[0].strip())
                vals = np.array([float(x.strip()) for x in lines[1:]])
                logger.info(f"Loading covariance matrix with header format (n={n})")
            except ValueError:
                # If first line is not an integer, assume it's all covariance values
                vals = np.array([float(x.strip()) for line in lines for x in line.split()])
                n = int(np.sqrt(len(vals)))
                logger.info(f"Loading covariance matrix without header, inferred n={n} from {len(vals)} values")

                # Verify it's a perfect square
                if n * n != len(vals):
                    raise ValueError(f"Covariance data has {len(vals)} values, which is not a perfect square")

            # Reshape to matrix
            cov = vals.reshape((n, n))

            # Validate covariance matrix
            if not np.allclose(cov, cov.T, rtol=1e-10):
                warnings.warn("Covariance matrix is not symmetric")
                cov = 0.5 * (cov + cov.T)  # Symmetrize

            # Check positive definiteness
            eigenvals = np.linalg.eigvals(cov)
            if np.any(eigenvals <= 0):
                warnings.warn("Covariance matrix is not positive definite")

            self._sn_cov = cov
            logger.info(f"Loaded {n}×{n} covariance matrix from {filename}")
            return cov

        except Exception as e:
            logger.error(f"Failed to load covariance matrix: {e}")
            raise  

    def load_hubble_data(self, filename: str = "Hz_all.dat") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load Hubble parameter measurements."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Hubble data file not found: {filepath}")
            
        try:
            data = np.loadtxt(filepath)
            z_obs, hub_obs, error_obs = data[:, 0], data[:, 1], data[:, 2]
            
            # Validate data
            if np.any(error_obs <= 0):
                raise ValueError("Non-positive errors found in Hubble data")
            
            self._hubble_data = (z_obs, hub_obs, error_obs)
            logger.info(f"Loaded {len(z_obs)} Hubble measurements from {filename}")
            return z_obs, hub_obs, error_obs
            
        except Exception as e:
            logger.error(f"Failed to load Hubble data: {e}")
            raise
    
    @property
    def sn_data(self) -> Optional[pd.DataFrame]:
        return self._sn_data
    
    @property
    def sn_cov(self) -> Optional[np.ndarray]:
        return self._sn_cov
    
    @property
    def hubble_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self._hubble_data


class PriorDistributions:
    """Class for handling prior distributions."""
    
    @staticmethod
    def uniform_w0(low: float = -2.0, high: float = -1.0) -> float:
        """Uniform prior for w0."""
        return uniform.rvs(loc=low, scale=high - low)
    
    @staticmethod
    def uniform_wa(low: float = -1.0, high: float = 1.0) -> float:
        """Uniform prior for wa."""
        return uniform.rvs(loc=low, scale=high - low)
    
    @staticmethod
    def gaussian_w0(mean: float = -1.0, std: float = 0.1) -> float:
        """Gaussian prior for w0."""
        return norm.rvs(loc=mean, scale=std)
    
    @staticmethod
    def gaussian_wa(mean: float = 0.0, std: float = 0.5) -> float:
        """Gaussian prior for wa."""
        return norm.rvs(loc=mean, scale=std)


class ABCRejectionSampler:
    """Approximate Bayesian Computation rejection sampler."""
    
    def __init__(self, cosmology: CosmologyCalculator, data_loader: DataLoader):
        self.cosmology = cosmology
        self.data_loader = data_loader
        self.priors = PriorDistributions()
        self.cov_inv = None  # Cache inverse covariance
    
    def chi2_distance(self, observed: np.ndarray, simulated: np.ndarray, 
                     cov_inv: np.ndarray) -> float:
        """
        Calculate chi-squared distance: (obs - sim)^T C^-1 (obs - sim).
        """
        try:
            delta = observed - simulated
            return np.dot(delta, np.dot(cov_inv, delta))
        except Exception as e:
            logger.error(f"Chi-squared calculation failed: {e}")
            return np.inf
    
    def simulate_sn_observables(self, w0: float, wa: float) -> Optional[np.ndarray]:
        """
        Simulate supernova distance moduli.
        """
        try:
            sn_data = self.data_loader.sn_data
            if sn_data is None:
                raise ValueError("SN data not loaded")
                
            z_values = sn_data['zCMB'].values
            mu_sim = self.cosmology.distance_modulus(z_values, w0, wa)
            
            # Check for invalid values
            if np.any(~np.isfinite(mu_sim)):
                return None
                
            return mu_sim
            
        except Exception as e:
            logger.debug(f"SN simulation failed for w0={w0:.3f}, wa={wa:.3f}: {e}")
            return None
    
    def simulate_hubble_observables(self, w0: float, wa: float) -> Optional[np.ndarray]:
        """
        Simulate Hubble parameter measurements.
        """
        try:
            hubble_data = self.data_loader.hubble_data
            if hubble_data is None:
                raise ValueError("Hubble data not loaded")
                
            z_obs = hubble_data[0]
            H_sim = self.cosmology.H0 * self.cosmology.hubble_normalized_cpl(z_obs, w0, wa)
            
            # Check for invalid values
            if np.any(~np.isfinite(H_sim)) or np.any(H_sim <= 0):
                return None
                
            return H_sim
            
        except Exception as e:
            logger.debug(f"Hubble simulation failed for w0={w0:.3f}, wa={wa:.3f}: {e}")
            return None
    
    def debug_distance_calculation(self, n_samples: int = 100) -> List[float]:
        """Debug function to check distance values."""
        logger.info("Running distance calculation debug...")
        
        sn_data = self.data_loader.sn_data
        sn_cov = self.data_loader.sn_cov
        
        if sn_data is None or sn_cov is None:
            raise ValueError("SN data and covariance matrix must be loaded first")
        
        mu_obs = sn_data['mb'].values
        
        # Cache inverse covariance
        if self.cov_inv is None:
            try:
                self.cov_inv = np.linalg.inv(sn_cov)
            except np.linalg.LinAlgError:
                logger.error("Covariance matrix is singular, using pseudoinverse")
                self.cov_inv = np.linalg.pinv(sn_cov)
        
        distances = []
        
        for i in range(n_samples):
            w0 = self.priors.uniform_w0(low=-2.0, high=-1.0)
            wa = self.priors.uniform_wa(low=-1.0, high=1.0)
            
            mu_sim = self.simulate_sn_observables(w0, wa)
            
            if mu_sim is not None:
                chi2 = self.chi2_distance(mu_obs, mu_sim, self.cov_inv)
                distances.append(chi2)
                
                if i % 10 == 0:
                    logger.info(f"Sample {i}: w0={w0:.3f}, wa={wa:.3f}, χ²={chi2:.1f}")
        
        if distances:
            logger.info(f"Distance statistics: min={np.min(distances):.1f}, "
                       f"max={np.max(distances):.1f}, mean={np.mean(distances):.1f}, "
                       f"median={np.median(distances):.1f}")
            
            # Check what percentage would be accepted with different tolerances
            tolerances = [100, 500, 1000, 5000, 10000]
            for tol in tolerances:
                accepted = sum(1 for d in distances if d <= tol)
                acceptance_rate = accepted / len(distances)
                logger.info(f"ε={tol}: {accepted}/{len(distances)} accepted "
                           f"({acceptance_rate*100:.2f}%)")
        else:
            logger.error("No valid simulations generated!")
        
        return distances
    
    def estimate_reasonable_tolerance(self, n_samples: int = 100) -> float:
        """Estimate a reasonable tolerance based on prior samples."""
        logger.info("Estimating reasonable tolerance...")
        
        sn_data = self.data_loader.sn_data
        sn_cov = self.data_loader.sn_cov
        
        if sn_data is None or sn_cov is None:
            raise ValueError("SN data and covariance matrix must be loaded first")
        
        mu_obs = sn_data['mb'].values
        
        # Cache inverse covariance
        if self.cov_inv is None:
            try:
                self.cov_inv = np.linalg.inv(sn_cov)
            except np.linalg.LinAlgError:
                logger.error("Covariance matrix is singular, using pseudoinverse")
                self.cov_inv = np.linalg.pinv(sn_cov)
        
        distances = []
        
        for i in range(n_samples):
            w0 = self.priors.uniform_w0(low=-2.0, high=-1.0)
            wa = self.priors.uniform_wa(low=-1.0, high=1.0)
            
            mu_sim = self.simulate_sn_observables(w0, wa)
            
            if mu_sim is not None:
                chi2 = self.chi2_distance(mu_obs, mu_sim, self.cov_inv)
                distances.append(chi2)
        
        if distances:
            # Accept best 10% of samples
            reasonable_tolerance = np.percentile(distances, 10)
            logger.info(f"Estimated reasonable tolerance: {reasonable_tolerance:.1f}")
            return reasonable_tolerance
        else:
            logger.warning("Could not estimate tolerance, using default 1000")
            return 1000.0
    
    def adaptive_rejection_sampling_sn(self, n_accepted: int, 
                                     initial_epsilon: float = None,
                                     max_iterations: int = 1000000) -> Tuple[np.ndarray, ...]:
        """
        Adaptive ABC rejection sampling that adjusts tolerance.
        """
        # First, estimate reasonable tolerance
        if initial_epsilon is None:
            initial_epsilon = self.estimate_reasonable_tolerance(n_samples=100)
        
        logger.info(f"Using adaptive tolerance, starting with ε={initial_epsilon:.1f}")
        
        sn_data = self.data_loader.sn_data
        sn_cov = self.data_loader.sn_cov
        
        if sn_data is None or sn_cov is None:
            raise ValueError("SN data and covariance matrix must be loaded first")
        
        mu_obs = sn_data['mb'].values
        
        # Cache inverse covariance
        if self.cov_inv is None:
            try:
                self.cov_inv = np.linalg.inv(sn_cov)
            except np.linalg.LinAlgError:
                logger.error("Covariance matrix is singular, using pseudoinverse")
                self.cov_inv = np.linalg.pinv(sn_cov)
        
        accepted_particles = []
        all_distances = []
        total_simulations = 0
        current_epsilon = initial_epsilon
        start_time = time.time()
        
        while len(accepted_particles) < n_accepted and total_simulations < max_iterations:
            # Sample from priors
            w0_sample = self.priors.uniform_w0(low=-2.0, high=-1.0)
            wa_sample = self.priors.uniform_wa(low=-1.0, high=1.0)
            
            # Simulate
            mu_sim = self.simulate_sn_observables(w0_sample, wa_sample)
            
            if mu_sim is not None:
                chi2 = self.chi2_distance(mu_obs, mu_sim, self.cov_inv)
                all_distances.append(chi2)
                
                # Accept if within current tolerance
                if chi2 <= current_epsilon:
                    accepted_particles.append((w0_sample, wa_sample, chi2))
                    
                    # Adapt tolerance if we have enough samples
                    if len(accepted_particles) >= 20:
                        current_epsilon = np.percentile([p[2] for p in accepted_particles], 75)
                    
                    if len(accepted_particles) % max(1, n_accepted // 10) == 0:
                        acceptance_rate = len(accepted_particles) / total_simulations
                        elapsed_time = time.time() - start_time
                        logger.info(f"Accepted {len(accepted_particles)}/{n_accepted}. "
                                  f"ε={current_epsilon:.1f}, rate={acceptance_rate:.4f}, "
                                  f"time={elapsed_time:.1f}s")
            
            total_simulations += 1
            
            if total_simulations % 10000 == 0:
                acceptance_rate = len(accepted_particles) / total_simulations
                elapsed_time = time.time() - start_time
                logger.info(f"Completed {total_simulations} simulations. "
                          f"Accepted: {len(accepted_particles)}. "
                          f"ε={current_epsilon:.1f}, rate={acceptance_rate:.6f}, "
                          f"time={elapsed_time:.1f}s")
        
        if len(accepted_particles) < n_accepted:
            logger.warning(f"Only accepted {len(accepted_particles)}/{n_accepted} particles "
                          f"after {total_simulations} simulations")
        
        final_acceptance_rate = len(accepted_particles) / total_simulations
        logger.info(f"ABC sampling completed. Final acceptance rate: {final_acceptance_rate:.6f}")
        
        # Extract results
        accepted_w0 = np.array([p[0] for p in accepted_particles])
        accepted_wa = np.array([p[1] for p in accepted_particles])
        accepted_chi2 = np.array([p[2] for p in accepted_particles])
        
        return accepted_w0, accepted_wa, accepted_chi2, total_simulations, np.array(all_distances)


class CosmologyDistances:
    """
    Cosmological distances for CPL dark energy model.
    """
    def __init__(self, Omega_m0=0.3, Omega_L0=0.7, h=0.71):
        self.Omega_m0 = Omega_m0
        self.Omega_L0 = Omega_L0
        self.h = h
        self.H0 = h * 100
        self.D_H = 2997.98 / h
        self._dL_interp = None
        self._zmax_interp = None

    def hubble_normalized_cpl(self, z, w0, wa):
        z = np.asarray(z)
        matter_term = self.Omega_m0 * (1 + z)**3
        de_exponent = 3 * (1 + w0 + wa)
        de_evolution = np.exp(-3 * wa * z / (1 + z))
        de_term = self.Omega_L0 * (1 + z)**de_exponent * de_evolution
        total = matter_term + de_term
        if np.any(total <= 0):
            raise ValueError("Negative energy density encountered. Check parameters.")
        return np.sqrt(total)

    def inverse_hubble_normalized_cpl(self, z, w0, wa):
        try:
            return 1.0 / self.hubble_normalized_cpl(z, w0, wa)
        except (ValueError, ZeroDivisionError):
            return np.inf

    def comoving_distance_cpl(self, z, w0, wa):
        if z <= 0:
            return 0.0
        integral, error = quad(
            lambda zp: self.inverse_hubble_normalized_cpl(zp, w0, wa),
            0, z, epsabs=1e-6, epsrel=1e-6, limit=100
        )
        return self.D_H * integral

    def luminosity_distance_cpl(self, z, w0, wa):
        if z <= 0:
            return 0.0
        return (1.0 + z) * self.comoving_distance_cpl(z, w0, wa)

    def build_luminosity_distance_interpolator(self, w0, wa, zmax=2.0, npoints=100):
        zgrid = np.linspace(1e-4, zmax, npoints)
        dLgrid = np.array([self.luminosity_distance_cpl(z, w0, wa) for z in zgrid])
        self._dL_interp = interp1d(zgrid, dLgrid, kind='cubic', bounds_error=False, fill_value='extrapolate')
        self._zmax_interp = zmax

    def luminosity_distance_fast(self, z, w0, wa):
        z = np.asarray(z)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)
        if (self._dL_interp is not None and self._zmax_interp is not None and np.all(z <= self._zmax_interp)):
            result = self._dL_interp(z)
        else:
            result = np.array([self.luminosity_distance_cpl(zi, w0, wa) for zi in z])
        return result.item() if scalar_input else result

    def distance_modulus(self, z, w0, wa):
        dL = self.luminosity_distance_fast(z, w0, wa)
        if np.any(dL <= 0) or np.any(~np.isfinite(dL)):
            return np.full_like(z, 1e6, dtype=float)
        return 5 * np.log10(dL) + 25

def simulate_pantheon_distance_modulus(w0, wa, Omega_m0=0.3, Omega_L0=0.7, h=0.71, data_path="binned_pantheon.txt"):
    """
    Simulate Pantheon binned SN distance modulus using CPL cosmology.
    """
    # Load Pantheon data
    df = pd.read_csv(data_path, sep=r'\s+')
    zcmb = df['zCMB'].values
    # Simulate distance modulus
    cosmo = CosmologyDistances(Omega_m0, Omega_L0, h)
    mu_sim = cosmo.distance_modulus(zcmb, w0, wa)
    return mu_sim

def load_pantheon_cov_matrix(cov_path="binned_cov_pantheon.txt"):
    """
    Load and reshape Pantheon binned covariance matrix.
    """
    with open(cov_path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    vals = np.array([float(x.strip()) for x in lines[1:]])
    cov = vals.reshape((n, n))
    return cov

# Example usage and main execution
def main():
    """Main function demonstrating the improved ABC sampling."""
    
    # Initialize components
    cosmology = CosmologyCalculator(Omega_m0=0.3, Omega_L0=0.7, h=0.71)
    data_loader = DataLoader(data_dir="/home/alfonsozapata/Documents/SimpleMC2/SimpleMC/simplemc/data")
    
    try:
        # Load data
        logger.info("Loading observational data...")
        data_loader.load_pantheon_data()
        data_loader.load_covariance_matrix()
        
        # Optional: Build interpolator for faster calculations
        logger.info("Building luminosity distance interpolator...")
        cosmology.build_luminosity_distance_interpolator(w0=-1.0, wa=0.0, zmax=3.0)
        
        # Initialize ABC sampler
        sampler = ABCRejectionSampler(cosmology, data_loader)
        
        # DEBUG: Check distance values first
        logger.info("Running debug distance calculation...")
        debug_distances = sampler.debug_distance_calculation(n_samples=100)
        
        # Test ΛCDM simulation
        logger.info("Testing LCDM simulation...")
        w0_lcdm, wa_lcdm = -1.0, 0.0
        mu_sim_lcdm = sampler.simulate_sn_observables(w0_lcdm, wa_lcdm)
        
        if mu_sim_lcdm is not None:
            sn_data = data_loader.sn_data
            mu_obs = sn_data['mb'].values
            
            # Get covariance inverse
            try:
                cov_inv = np.linalg.inv(data_loader.sn_cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(data_loader.sn_cov)
                
            chi2_lcdm = sampler.chi2_distance(mu_obs, mu_sim_lcdm, cov_inv)
            logger.info(f"LCDM (w0=-1, wa=0) χ² = {chi2_lcdm:.1f}")
        else:
            logger.error("LCDM simulation failed!")
        
        # If distances are reasonable, run ABC
        if debug_distances and np.min(debug_distances) < 10000:  # More lenient threshold
            logger.info("Distances look reasonable, proceeding with ABC...")
            
            # Run adaptive ABC rejection sampling
            n_accepted = 100
            
            results = sampler.adaptive_rejection_sampling_sn(
                n_accepted=n_accepted,
                max_iterations=100000
            )
            
            accepted_w0, accepted_wa, accepted_chi2, total_sim, all_distances = results
            
            # Print results
            print(f"\nABC Results Summary:")
            print(f"Total simulations: {total_sim}")
            print(f"Accepted particles: {len(accepted_w0)}")
            print(f"Final acceptance rate: {len(accepted_w0)/total_sim:.6f}")
            print(f"\nParameter Statistics:")
            print(f"w0: mean = {np.mean(accepted_w0):.3f}, std = {np.std(accepted_w0):.3f}")
            print(f"wa: mean = {np.mean(accepted_wa):.3f}, std = {np.std(accepted_wa):.3f}")
            print(f"χ²: mean = {np.mean(accepted_chi2):.1f}, min = {np.min(accepted_chi2):.1f}")
            
            # Plot results
            if len(accepted_w0) > 0:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.scatter(accepted_w0, accepted_wa, c=accepted_chi2, cmap='viridis', alpha=0.7)
                plt.colorbar(label='χ²')
                plt.axvline(-1, color='red', linestyle='--', alpha=0.7, label='ΛCDM (w₀=-1)')
                plt.axhline(0, color='red', linestyle='--', alpha=0.7, label='ΛCDM (wₐ=0)')
                plt.xlabel('w₀')
                plt.ylabel('wₐ')
                plt.title('Accepted Particles')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.hist(accepted_chi2, bins=20, alpha=0.7, color='orange', edgecolor='black')
                plt.xlabel('χ²')
                plt.ylabel('Count')
                plt.title('Distance Distribution of Accepted Particles')
                
                plt.tight_layout()
                plt.show()
            
        else:
            logger.error("Distance values are too large. Check simulation and data.")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()