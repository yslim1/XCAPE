"""
Utility functions for PXRD pattern processing and comparison.
"""

from typing import Tuple, List, Optional, Union

import numpy as np
from scipy.stats import wasserstein_distance
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# ==============================================================
# Constants
# ==============================================================

# Peak detection and pattern generation parameters
PEAK_THRESHOLD_RATIO = 0.01  # 1% of max intensity for binary peak detection
DEFAULT_STEP_SIZE = 0.02     # degrees, grid resolution for Gaussian patterns
DEFAULT_FWHM = 0.25          # degrees, full width at half maximum for Gaussian peaks
DEFAULT_PEAK_TOL = 0.15      # degrees, tolerance for peak matching

# XRD wavelengths in angstroms
# from pymatgen
WAVELENGTHS = {
    "CuKa": 1.54184,
    "CuKa2": 1.54439,
    "CuKa1": 1.54056,
    "CuKb1": 1.39222,
    "MoKa": 0.71073,
    "MoKa2": 0.71359,
    "MoKa1": 0.70930,
    "MoKb1": 0.63229,
    "CrKa": 2.29100,
    "CrKa2": 2.29361,
    "CrKa1": 2.28970,
    "CrKb1": 2.08487,
    "FeKa": 1.93735,
    "FeKa2": 1.93998,
    "FeKa1": 1.93604,
    "FeKb1": 1.75661,
    "CoKa": 1.79026,
    "CoKa2": 1.79285,
    "CoKa1": 1.78896,
    "CoKb1": 1.63079,
    "AgKa": 0.560885,
    "AgKa2": 0.563813,
    "AgKa1": 0.559421,
    "AgKb1": 0.497082,
}


# ==============================================================
# Helper Functions
# ==============================================================

def _normalize_intensities(intensities: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Normalize intensity array by specified method.
    
    Args:
        intensities: Array of intensity values
        method: Normalization method - None, 'max', or 'sum'
    
    Returns:
        Normalized intensity array
    
    Raises:
        ValueError: If method is not None, 'max', or 'sum'
    """
    if intensities.size == 0 or method is None:
        return intensities
    
    if method == 'max':
        max_val = intensities.max()
        return intensities / max_val if max_val > 0 else intensities
    
    if method == 'sum':
        sum_val = intensities.sum()
        return intensities / sum_val if sum_val > 0 else intensities
    
    raise ValueError("normalize must be None, 'max', or 'sum'")


# ==============================================================
# XRD Pattern Comparison Functions
# ==============================================================

def gaussian_xrd(
    two_theta_peaks: List[float],
    intensities: List[float],
    t_range: Tuple[float, float],
    step: float = DEFAULT_STEP_SIZE,
    fwhm: float = DEFAULT_FWHM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Gaussian-smeared XRD pattern from peak positions.
    
    Args:
        two_theta_peaks: Peak positions in 2θ degrees
        intensities: Peak intensities
        t_range: (min, max) 2θ range in degrees
        step: Grid step size in degrees (default: 0.02)
        fwhm: Full width at half maximum for Gaussian peaks (default: 0.25)
    
    Returns:
        Tuple of (grid, pattern) where pattern is normalized to max=1 if possible
    """
    two_theta_peaks = np.asarray(two_theta_peaks, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    # Drop non-finite peaks so a single NaN/inf position or intensity cannot
    # poison the entire broadened curve (NaN propagates through the Gaussian
    # sum, making the pattern -- and every downstream metric -- NaN). Guarded on
    # equal shapes so mismatched inputs keep the historical zip-truncation path.
    if two_theta_peaks.shape == intensities.shape and two_theta_peaks.size:
        finite = np.isfinite(two_theta_peaks) & np.isfinite(intensities)
        two_theta_peaks = two_theta_peaks[finite]
        intensities = intensities[finite]

    theta_min, theta_max = float(t_range[0]), float(t_range[1])
    if theta_max <= theta_min:
        raise ValueError("t_range max must be greater than min")

    grid = np.arange(theta_min, theta_max + 1e-12, step, dtype=float)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    pattern = np.zeros_like(grid)

    if two_theta_peaks.size:
        # Accumulate one Gaussian per peak onto the shared grid.
        for t0, I in zip(two_theta_peaks, intensities):
            pattern += I * np.exp(-0.5 * ((grid - float(t0)) / sigma) ** 2)

        max_val = pattern.max()
        if max_val > 0:
            pattern = pattern / max_val

    return grid, pattern


def r_factor_peak_pairing(
    two_theta_obs: List[float],
    I_obs: List[float],
    two_theta_calc: List[float],
    I_calc: List[float],
    tol: float = DEFAULT_PEAK_TOL,
    normalize: Optional[str] = None,  # None | 'max' | 'sum' | 'scale'
    two_theta_range: Optional[Tuple[float, float]] = None,
    return_pairs: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """Compute R-factor by pairing each calculated peak to its nearest observed peak.

    Pairing follows the scheme used by Mint (github.com/materials/mint,
    Diffraction::matchPeaksToReference / getCurrentRFactor): each calculated
    peak is matched to its single nearest observed peak within `tol` (ties
    broken toward the lower-angle candidate); calculated peaks with no
    observed peak within `tol` are unmatched and counted as pure error.
    Observed peaks are fixed reference bins and are never merged with each
    other, so two genuinely distinct observed peaks always stay separate even
    if a calculated peak happens to sit between them -- this is what keeps a
    calculated pattern that is missing a real reflection from hiding that gap
    by letting a neighboring peak absorb it into a shared comparison bin.

    Args:
        two_theta_obs: Observed peak positions in degrees
        I_obs: Observed intensities
        two_theta_calc: Calculated peak positions in degrees
        I_calc: Calculated intensities
        tol: Matching tolerance in degrees (default: 0.15)
        normalize: Normalization method - None, 'max', 'sum', or 'scale' (default: None).
        two_theta_range: Optional (min, max) 2θ range in degrees
        return_pairs: If True, return (R, pairs) instead of just R

    Returns:
        R-factor (float) or tuple of (R-factor, pairs array) if return_pairs is True.
        Pairs array has columns: [two_theta, Iobs, Icalc_matched] -- one row
        per observed peak (Icalc_matched summed from any calculated peaks
        matched to it, 0.0 if none), plus one row per unmatched calculated
        peak (Iobs=0.0), sorted by two_theta.
    """
    two_theta_obs = np.asarray(two_theta_obs, dtype=float)
    I_obs = np.asarray(I_obs, dtype=float)
    two_theta_calc = np.asarray(two_theta_calc, dtype=float)
    I_calc = np.asarray(I_calc, dtype=float)

    # restrict to range
    if two_theta_range is not None:
        t_min, t_max = two_theta_range
        mask_obs = (two_theta_obs >= t_min) & (two_theta_obs <= t_max)
        two_theta_obs = two_theta_obs[mask_obs]
        I_obs = I_obs[mask_obs]
        mask_calc = (two_theta_calc >= t_min) & (two_theta_calc <= t_max)
        two_theta_calc = two_theta_calc[mask_calc]
        I_calc = I_calc[mask_calc]

    # Drop non-finite peaks: d_to_two_theta emits NaN for unreachable
    # reflections (sin(theta) > 1), and NaN sorts to the array end, silently
    # corrupting the nearest-peak search rather than raising. Keep only peaks
    # whose position and intensity are both finite.
    finite_obs = np.isfinite(two_theta_obs) & np.isfinite(I_obs)
    two_theta_obs = two_theta_obs[finite_obs]
    I_obs = I_obs[finite_obs]
    finite_calc = np.isfinite(two_theta_calc) & np.isfinite(I_calc)
    two_theta_calc = two_theta_calc[finite_calc]
    I_calc = I_calc[finite_calc]

    if two_theta_obs.size == 0 and two_theta_calc.size == 0:
        return (0.0, np.empty((0, 3))) if return_pairs else 0.0

    # Observed peaks are fixed bins: sort once, never merged with each other.
    obs_order = np.argsort(two_theta_obs)
    two_theta_obs = two_theta_obs[obs_order]
    I_obs = I_obs[obs_order]

    matched_calc_sum = np.zeros_like(two_theta_obs)
    unmatched_mask = np.ones(two_theta_calc.shape, dtype=bool)

    if two_theta_obs.size > 0 and two_theta_calc.size > 0:
        # Nearest observed-peak index for each calculated peak: binary search
        # over the sorted observed positions, then compare the two bracketing
        # candidates (the insertion point and its predecessor).
        insert_idx = np.searchsorted(two_theta_obs, two_theta_calc)
        idx_lo = np.clip(insert_idx - 1, 0, two_theta_obs.size - 1)
        idx_hi = np.clip(insert_idx, 0, two_theta_obs.size - 1)
        dist_lo = np.abs(two_theta_calc - two_theta_obs[idx_lo])
        dist_hi = np.abs(two_theta_calc - two_theta_obs[idx_hi])
        nearest_idx = np.where(dist_lo <= dist_hi, idx_lo, idx_hi)
        nearest_dist = np.minimum(dist_lo, dist_hi)

        unmatched_mask = nearest_dist > tol
        matched_idx = nearest_idx[~unmatched_mask]
        # Multiple calculated peaks may share the same nearest observed bin
        # (e.g. a Kalpha1/Kalpha2 doublet); sum their intensities into it.
        np.add.at(matched_calc_sum, matched_idx, I_calc[~unmatched_mask])

    obs_rows = np.column_stack([two_theta_obs, I_obs, matched_calc_sum])
    unmatched_rows = np.column_stack([
        two_theta_calc[unmatched_mask],
        np.zeros(int(unmatched_mask.sum())),
        I_calc[unmatched_mask],
    ])

    pairs = np.vstack([obs_rows, unmatched_rows])
    pairs = pairs[np.argsort(pairs[:, 0])]

    if normalize == 'scale':
        # Least-squares scale fit (Mint's DR_SQUARED): minimizes
        # sum((Iobs - scale*Icalc)^2), i.e. scale = <Iobs,Icalc> / <Icalc,Icalc>.
        # Iobs is left unscaled; only the matched-calc column is rescaled.
        calc_sq_sum = np.sum(pairs[:, 2] ** 2)
        scale = float(np.sum(pairs[:, 1] * pairs[:, 2]) / calc_sq_sum) if calc_sq_sum > 0 else 1.0
        pairs[:, 2] = scale * pairs[:, 2]
    else:
        # Normalize obs/matched-calc columns (post-pairing), each independently
        pairs[:, 1] = _normalize_intensities(pairs[:, 1], normalize)
        pairs[:, 2] = _normalize_intensities(pairs[:, 2], normalize)

    diff2 = (pairs[:, 2] - pairs[:, 1]) ** 2
    denom = np.sum(pairs[:, 1] ** 2)
    R = float(np.sum(diff2) / denom) if denom > 0 else float('inf')

    return (R, pairs) if return_pairs else R


def d_to_two_theta(d_angstrom: List[float], wavelength: str) -> np.ndarray:
    """Convert d-spacing to 2θ degrees using Bragg's law.
    
    Args:
        d_angstrom: d-spacing values in angstroms
        wavelength: X-ray wavelength key (e.g., 'CuKa', 'MoKa')
    
    Returns:
        Array of 2θ values in degrees
    """
    d_angstrom = np.asarray(d_angstrom, dtype=float)
    ratio = WAVELENGTHS[wavelength] / (2.0 * d_angstrom)
    valid = ratio <= 1.0
    theta = np.full_like(d_angstrom, np.nan)
    theta[valid] = np.arcsin(ratio[valid])
    two_theta = np.degrees(2.0 * theta)
    return two_theta


def generate_pymatgen_pxrd_pattern(structure, two_theta_range: Optional[Tuple[float, float]] = None, wavelength: str = "CuKa") -> Tuple[np.ndarray, np.ndarray]:
    """Generate PXRD pattern using pymatgen's XRDCalculator.
    
    Args:
        structure: Crystal structure (convertible to pymatgen structure)
        two_theta_range: Optional (min, max) 2θ range in degrees (default: (0, 90))
        wavelength: X-ray wavelength key (default: 'CuKa')
    
    Returns:
        Tuple of (two_theta, intensities) arrays
    """
    xrd_calculator = XRDCalculator(wavelength=wavelength)
    if two_theta_range is None:
        two_theta_range = (0.0, 90.0)
    pmg_structure = structure.to_pymatgen()
    pattern = xrd_calculator.get_pattern(structure=pmg_structure, two_theta_range=two_theta_range)
    return np.array(pattern.x), np.array(pattern.y)


def wasserstein_dist(x_obs: List[float], y_obs: List[float], x_calc: List[float], y_calc: List[float], two_theta_range: Tuple[float, float]) -> float:
    """Compute Wasserstein distance between two XRD patterns.

    Converts each peak list to a Gaussian-smeared pattern on a shared 2θ grid,
    then computes the 1D Wasserstein distance treating intensities as probability weights.
    The result is in degrees (weighted average 2θ shift to transform one pattern into the other).
    scipy normalizes weights by their sum internally, so only the relative shape matters.

    Args:
        x_obs: Observed peak positions in 2θ degrees
        y_obs: Observed intensities
        x_calc: Calculated peak positions in 2θ degrees
        y_calc: Calculated intensities
        two_theta_range: (min, max) 2θ range in degrees

    Returns:
        Wasserstein distance in degrees (lower is better)
    """
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    x_calc = np.asarray(x_calc, dtype=float)
    y_calc = np.asarray(y_calc, dtype=float)

    # Generate Gaussian-smeared patterns on a shared 2θ grid (already max-normalized)
    grid, pattern_obs  = gaussian_xrd(x_obs,  y_obs,  two_theta_range,
                                      step=DEFAULT_STEP_SIZE, fwhm=DEFAULT_FWHM)
    _,    pattern_calc = gaussian_xrd(x_calc, y_calc, two_theta_range,
                                      step=DEFAULT_STEP_SIZE, fwhm=DEFAULT_FWHM)

    if pattern_obs.sum() == 0 or pattern_calc.sum() == 0:
        return float('inf')

    # Grid is in degrees; scipy normalizes weights internally
    distance = wasserstein_distance(grid, grid, pattern_obs, pattern_calc)
    return float(distance)

__all__ = [
    'gaussian_xrd',
    'r_factor_peak_pairing',
    'wasserstein_dist',
    'd_to_two_theta',
    'generate_pymatgen_pxrd_pattern',
]
