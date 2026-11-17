import numpy as np
import pytest

from xcape.utils import xrd_utils


# =============================================================================
# gaussian_xrd
# =============================================================================
def test_gaussian_xrd_grid_covers_range_and_peak_is_normalized():
    grid, pattern = xrd_utils.gaussian_xrd([10.0], [50.0], (5.0, 15.0), step=0.5, fwhm=0.5)
    assert grid[0] == pytest.approx(5.0)
    assert grid[-1] == pytest.approx(15.0)
    assert pattern.max() == pytest.approx(1.0)
    # Peak of the single Gaussian should sit at the requested 2theta.
    assert grid[np.argmax(pattern)] == pytest.approx(10.0)


def test_gaussian_xrd_empty_peaks_returns_zero_pattern():
    grid, pattern = xrd_utils.gaussian_xrd([], [], (0.0, 10.0), step=1.0)
    assert np.all(pattern == 0.0)
    assert len(grid) == len(pattern)


def test_gaussian_xrd_invalid_range_raises():
    with pytest.raises(ValueError):
        xrd_utils.gaussian_xrd([1.0], [1.0], (10.0, 5.0))


def test_gaussian_xrd_drops_nonfinite_peaks():
    # A NaN peak position (e.g. an unreachable reflection from d_to_two_theta)
    # must not poison the whole broadened pattern.
    grid, pattern = xrd_utils.gaussian_xrd(
        [10.0, np.nan], [1.0, 0.5], (5.0, 15.0), step=0.5, fwhm=0.5
    )
    assert np.all(np.isfinite(pattern))
    assert pattern.max() > 0.0
    # Result matches the pattern with the NaN peak simply omitted.
    _, expected = xrd_utils.gaussian_xrd([10.0], [1.0], (5.0, 15.0), step=0.5, fwhm=0.5)
    np.testing.assert_allclose(pattern, expected)


# =============================================================================
# r_factor_peak_pairing
# =============================================================================
def test_r_factor_identical_patterns_is_zero():
    two_theta = [10.0, 20.0, 30.0]
    intensity = [100.0, 50.0, 20.0]
    R = xrd_utils.r_factor_peak_pairing(two_theta, intensity, two_theta, intensity)
    assert R == pytest.approx(0.0)


def test_r_factor_no_peaks_returns_zero():
    assert xrd_utils.r_factor_peak_pairing([], [], [], []) == 0.0


def test_r_factor_no_peaks_return_pairs_shape():
    R, pairs = xrd_utils.r_factor_peak_pairing([], [], [], [], return_pairs=True)
    assert R == 0.0
    assert pairs.shape == (0, 3)


def test_r_factor_calc_peak_within_tolerance_matches_nearest_obs_peak():
    # obs and calc peaks 0.1 deg apart (tol default 0.15) must match.
    R, pairs = xrd_utils.r_factor_peak_pairing(
        [10.0], [100.0], [10.1], [90.0], return_pairs=True,
    )
    assert pairs.shape == (1, 3)
    # Row is centered on the observed peak's own position, not a midpoint --
    # observed peaks are fixed reference bins, never averaged/merged.
    assert pairs[0, 0] == pytest.approx(10.0)
    assert pairs[0, 1] == pytest.approx(100.0)
    assert pairs[0, 2] == pytest.approx(90.0)
    expected_R = (90.0 - 100.0) ** 2 / (100.0 ** 2)
    assert R == pytest.approx(expected_R)


def test_r_factor_peaks_outside_tolerance_are_unmatched():
    R, pairs = xrd_utils.r_factor_peak_pairing(
        [10.0], [100.0], [20.0], [90.0], tol=0.15, return_pairs=True,
    )
    assert pairs.shape == (2, 3)
    # obs bin (no matching calc peak) then unmatched calc peak, by 2theta.
    assert pairs[0, 1] == pytest.approx(100.0)
    assert pairs[0, 2] == pytest.approx(0.0)
    assert pairs[1, 1] == pytest.approx(0.0)
    assert pairs[1, 2] == pytest.approx(90.0)


def test_r_factor_observed_peaks_are_never_merged_with_each_other():
    # Two obs peaks close together with no calc peaks at all: each must stay
    # its own row -- observed peaks are fixed bins, never merged, regardless
    # of how close together they are.
    R, pairs = xrd_utils.r_factor_peak_pairing(
        [10.0, 10.1], [50.0, 60.0], [], [], tol=0.15, return_pairs=True,
    )
    assert pairs.shape == (2, 3)


def test_r_factor_bridging_calc_peak_does_not_merge_distinct_obs_peaks():
    # Two distinct obs peaks (0.28 apart, i.e. further apart than tol) with a
    # single calc peak roughly between them: this simulates a calculated
    # pattern that is missing one real reflection. The calc peak may only
    # match ONE of the two obs peaks (whichever is nearer, ties broken to the
    # lower-angle one); the other obs peak must show up as a fully unmatched
    # reference bin (Icalc=0), not silently absorbed into a merged cluster.
    R, pairs = xrd_utils.r_factor_peak_pairing(
        [10.0, 10.28], [100.0, 100.0], [10.14], [100.0],
        tol=0.15, return_pairs=True,
    )
    assert pairs.shape == (2, 3)
    matched_row = pairs[pairs[:, 2] > 0]
    unmatched_row = pairs[pairs[:, 2] == 0]
    assert matched_row.shape == (1, 3)
    assert unmatched_row.shape == (1, 3)
    # The unmatched obs peak contributes its full intensity as error, so R
    # must be twice what a (buggy) merge-everything-together scheme would give.
    assert R == pytest.approx(0.5)


def test_r_factor_two_theta_range_filters_out_of_range_peaks():
    R = xrd_utils.r_factor_peak_pairing(
        [10.0, 50.0], [100.0, 999.0], [10.0, 60.0], [100.0, 999.0],
        two_theta_range=(0.0, 20.0),
    )
    assert R == pytest.approx(0.0)


@pytest.mark.parametrize('normalize,expected', [('max', 1.0), ('sum', 100.0 / 150.0)])
def test_r_factor_normalize_bounds_matched_obs_row(normalize, expected):
    # obs peaks at 10.0 (I=100) and 10.05 (I=50); calc peak at 10.02 (I=80) is
    # nearer to 10.0 (dist 0.02) than to 10.05 (dist 0.03), so it matches only
    # the first obs bin.
    R, pairs = xrd_utils.r_factor_peak_pairing(
        [10.0, 10.05], [100.0, 50.0], [10.02], [80.0],
        normalize=normalize, return_pairs=True,
    )
    assert pairs[0, 1] == pytest.approx(expected)
    # The matched calc column is normalized independently, so its sole
    # nonzero entry always hits the normalization bound (1.0) regardless.
    assert pairs[0, 2] == pytest.approx(1.0)


def test_r_factor_invalid_normalize_raises():
    with pytest.raises(ValueError):
        xrd_utils.r_factor_peak_pairing([10.0], [1.0], [10.0], [1.0], normalize='bogus')


def test_r_factor_normalize_scale_fits_least_squares_optimal_multiplier():
    # calc is exactly half of obs at every matched peak, so the least-squares
    # optimal scale is exactly 2.0 and R should drop to 0 after fitting it.
    R, pairs = xrd_utils.r_factor_peak_pairing(
        [10.0, 20.0], [100.0, 50.0], [10.0, 20.0], [50.0, 25.0],
        normalize='scale', return_pairs=True,
    )
    # obs column is left unscaled by 'scale' normalization.
    assert pairs[:, 1] == pytest.approx([100.0, 50.0])
    assert pairs[:, 2] == pytest.approx([100.0, 50.0])
    assert R == pytest.approx(0.0)


def test_r_factor_normalize_scale_with_no_calc_peaks_does_not_divide_by_zero():
    R = xrd_utils.r_factor_peak_pairing(
        [10.0], [100.0], [], [], normalize='scale',
    )
    assert R == pytest.approx(1.0)


# =============================================================================
# d_to_two_theta
# =============================================================================
def test_d_to_two_theta_known_value():
    # Bragg's law: 2 d sin(theta) = wavelength => theta = arcsin(wavelength / 2d)
    wavelength = xrd_utils.WAVELENGTHS['CuKa']
    d = wavelength  # sin(theta) = 0.5 -> theta = 30 deg -> two_theta = 60
    two_theta = xrd_utils.d_to_two_theta([d], wavelength='CuKa')
    assert two_theta[0] == pytest.approx(60.0)


def test_d_to_two_theta_out_of_range_is_nan():
    # Very small d-spacing makes wavelength / (2d) > 1, which is unphysical.
    two_theta = xrd_utils.d_to_two_theta([0.01], wavelength='CuKa')
    assert np.isnan(two_theta[0])


# =============================================================================
# wasserstein_dist
# =============================================================================
def test_wasserstein_dist_identical_patterns_is_zero():
    two_theta = [10.0, 20.0]
    intensity = [100.0, 50.0]
    dist = xrd_utils.wasserstein_dist(
        two_theta, intensity, two_theta, intensity, two_theta_range=(0.0, 30.0)
    )
    assert dist == pytest.approx(0.0, abs=1e-6)


def test_wasserstein_dist_shifted_pattern_is_positive():
    dist = xrd_utils.wasserstein_dist(
        [10.0], [100.0], [15.0], [100.0], two_theta_range=(0.0, 30.0)
    )
    assert dist > 0.0


def test_wasserstein_dist_empty_pattern_is_inf():
    dist = xrd_utils.wasserstein_dist([], [], [10.0], [100.0], two_theta_range=(0.0, 30.0))
    assert dist == float('inf')


def test_wasserstein_dist_finite_with_nonfinite_reference_peak():
    # A non-finite reference peak (unreachable reflection) must not make the
    # Wasserstein distance NaN; the finite reference peak should still drive it
    # to ~0 against an identical observed pattern.
    import math

    dist = xrd_utils.wasserstein_dist(
        [10.0, np.nan], [100.0, 50.0], [10.0], [100.0], two_theta_range=(0.0, 30.0)
    )
    assert math.isfinite(dist)
    assert dist == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# generate_pymatgen_pxrd_pattern
# =============================================================================
def test_generate_pymatgen_pxrd_pattern_returns_peaks_within_range(nacl_pyxtal):
    # generate_pymatgen_pxrd_pattern expects a PyXtal-like object with
    # .to_pymatgen(), not a pymatgen Structure directly.
    two_theta, intensity = xrd_utils.generate_pymatgen_pxrd_pattern(
        nacl_pyxtal, two_theta_range=(0.0, 90.0),
    )
    assert len(two_theta) == len(intensity)
    assert len(two_theta) > 0
    assert np.all(two_theta >= 0.0) and np.all(two_theta <= 90.0)
