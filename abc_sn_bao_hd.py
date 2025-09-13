import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Tuple, Optional, Callable, Union
import warnings
import logging

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
        
        Parameters:
        -----------
        Omega_m0 : float
            Matter density parameter at z=0
        Omega_L0 : float 
            Dark energy density parameter at z=0
        h : float
            Dimensionless Hubble parameter
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
        
        Parameters:
        -----------
        z : float or array
            Redshift
        w0 : float
            Present dark energy equation of state
        wa : float
            Evolution parameter for dark energy equation of state
            
        Returns:
        --------
        H(z)/H0 : float or array
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
        
        Parameters:
        -----------
        z : float
            Redshift
        w0, wa : float
            CPL parameters
            
        Returns:
        --------
        d_c : float
            Comoving distance in Mpc
        """
        if z <= 0:
            return 0.0
            
        try:
            integral, error = quad(
                lambda zp: self.inverse_hubble_normalized_cpl(zp, w0, wa),
                0, z, 
                epsabs=1e-8, 
                epsrel=1e-8, 
                limit=500
            )
            
            # Check integration error
            if error > 1e-6 * integral:
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
                                             zmax: float = 2.0, npoints: int = 200):
        """
        Build interpolator for faster luminosity distance calculations.
        
        Parameters:
        -----------
        w0, wa : float
            CPL parameters
        zmax : float
            Maximum redshift for interpolation
        npoints : int
            Number of interpolation points
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
        
        Parameters:
        -----------
        z : float or array
            Redshift(s)
        w0, wa : float
            CPL parameters
            
        Returns:
        --------
        d_L : float or array
            Luminosity distance(s) in Mpc
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
        
        Parameters:
        -----------
        z : float or array
            Redshift(s)
        w0, wa : float
            CPL parameters
            
        Returns:
        --------
        mu : float or array
            Distance modulus
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
    
    def load_pantheon_data(self, filename: str = "Pantheon+SH0ES.dat") -> pd.DataFrame:
        """Load Pantheon+SH0ES supernova data."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        try:
            data = pd.read_csv(filepath, sep=r'\s+')
            required_cols = ['zCMB', 'MU_SH0ES']
            
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")
                
            self._sn_data = data
            logger.info(f"Loaded {len(data)} SN data points from {filename}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load Pantheon data: {e}")
            raise
    
    def load_covariance_matrix(self, filename: str = "Pantheon+SH0ES_STAT+SYS.cov") -> np.ndarray:
        """Load covariance matrix for SN data."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Covariance file not found: {filepath}")
            
        try:
            with open(filepath) as f:
                lines = f.readlines()
            
            n = int(lines[0].strip())
            vals = np.array([float(x.strip()) for x in lines[1:]])
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
    
    def chi2_distance(self, observed: np.ndarray, simulated: np.ndarray, 
                     cov_inv: np.ndarray) -> float:
        """
        Calculate chi-squared distance: (obs - sim)^T C^-1 (obs - sim).
        
        Parameters:
        -----------
        observed : array
            Observed data
        simulated : array  
            Simulated data
        cov_inv : array
            Inverse covariance matrix
            
        Returns:
        --------
        chi2 : float
            Chi-squared value
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
        
        Parameters:
        -----------
        w0, wa : float
            CPL parameters
            
        Returns:
        --------
        mu_sim : array or None
            Simulated distance moduli (None if simulation failed)
        """
        try:
            sn_data = self.data_loader.sn_data
            if sn_data is None:
                raise ValueError("SN data not loaded")
                
            zcmb = sn_data['zCMB'].values
            mu_sim = self.cosmology.distance_modulus(zcmb, w0, wa)
            
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
        
        Parameters:
        -----------
        w0, wa : float
            CPL parameters
            
        Returns:
        --------
        H_sim : array or None
            Simulated Hubble parameters (None if simulation failed)
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
    
    def rejection_sampling_sn(self, n_accepted: int, epsilon: float,
                            prior_w0: Callable = None, prior_wa: Callable = None,
                            max_iterations: int = None) -> Tuple[np.ndarray, ...]:
        """
        ABC rejection sampling for supernova data.
        
        Parameters:
        -----------
        n_accepted : int
            Number of particles to accept
        epsilon : float
            Tolerance threshold
        prior_w0, prior_wa : callable
            Prior sampling functions
        max_iterations : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple of (accepted_w0, accepted_wa, accepted_chi2, total_simulations, all_distances)
        """
        # Set default priors
        if prior_w0 is None:
            prior_w0 = self.priors.uniform_w0
        if prior_wa is None:
            prior_wa = self.priors.uniform_wa
        
        if max_iterations is None:
            max_iterations = n_accepted * 10000  # Reasonable default
        
        # Load data
        sn_data = self.data_loader.sn_data
        sn_cov = self.data_loader.sn_cov
        
        if sn_data is None or sn_cov is None:
            raise ValueError("SN data and covariance matrix must be loaded first")
        
        mu_obs = sn_data['MU_SH0ES'].values
        
        try:
            cov_inv = np.linalg.inv(sn_cov)
        except np.linalg.LinAlgError:
            logger.error("Covariance matrix is singular, using pseudoinverse")
            cov_inv = np.linalg.pinv(sn_cov)
        
        # Initialize storage
        accepted_particles = []
        all_distances = []
        total_simulations = 0
        
        logger.info(f"Starting ABC rejection sampling for SN data")
        logger.info(f"Target: {n_accepted} particles, tolerance: eps = {epsilon}")
        
        while len(accepted_particles) < n_accepted and total_simulations < max_iterations:
            # Sample from priors
            w0_sample = prior_w0()
            wa_sample = prior_wa()
            
            # Simulate observables
            mu_sim = self.simulate_sn_observables(w0_sample, wa_sample)
            
            if mu_sim is not None:
                # Calculate distance
                chi2 = self.chi2_distance(mu_obs, mu_sim, cov_inv)
                all_distances.append(chi2)
                
                # Accept/reject
                if chi2 <= epsilon:
                    accepted_particles.append((w0_sample, wa_sample, chi2))
                    
                    if len(accepted_particles) % max(1, n_accepted // 10) == 0:
                        acceptance_rate = len(accepted_particles) / total_simulations
                        logger.info(f"Accepted {len(accepted_particles)}/{n_accepted} particles. "
                                  f"Acceptance rate: {acceptance_rate:.4f}")
            
            total_simulations += 1
            
            # Progress update
            if total_simulations % 10000 == 0:
                acceptance_rate = len(accepted_particles) / total_simulations
                logger.info(f"Completed {total_simulations} simulations. "
                          f"Accepted: {len(accepted_particles)}. "
                          f"Acceptance rate: {acceptance_rate:.6f}")
        
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


# Example usage and main execution
def main():
    """Main function demonstrating the improved ABC sampling."""
    
    # Initialize components
    cosmology = CosmologyCalculator(Omega_m0=0.3, Omega_L0=0.7, h=0.71)
    data_loader = DataLoader(data_dir="/Users/alfonsozapata/Documents/SimpleMC/simplemc/data/")
    
    try:
        # Load data
        logger.info("Loading observational data...")
        data_loader.load_pantheon_data()
        data_loader.load_covariance_matrix()
        
        # Optional: Build interpolator for faster calculations
        logger.info("Building luminosity distance interpolator...")
        cosmology.build_luminosity_distance_interpolator(w0=-1.0, wa=0.0, zmax=2.0)
        
        # Initialize ABC sampler
        sampler = ABCRejectionSampler(cosmology, data_loader)
        
        # Run ABC rejection sampling
        logger.info("Starting ABC rejection sampling...")
        epsilon_sn = 1000.0  # Tolerance
        n_accepted = 50    # Number of accepted particles
        
        results = sampler.rejection_sampling_sn(
            n_accepted=n_accepted,
            epsilon=epsilon_sn,
            prior_w0=lambda: uniform.rvs(loc=-2.0, scale=1.0),  # w0 ∈ [-2, -1]
            prior_wa=lambda: uniform.rvs(loc=-1.0, scale=2.0),  # wa ∈ [-1, 1]
            max_iterations=1000000
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
       # print(f"²: mean = {np.mean(accepted_chi2):.1f}, min = {np.min(accepted_chi2):.1f}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()  