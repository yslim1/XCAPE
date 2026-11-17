"""Post-processing module for XCAPE structure generation.

This module handles XRD metric calculation, candidate selection,
deduplication, and output file writing after structure relaxation.
"""
from typing import Dict, Any, List, Tuple
from pathlib import Path

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

from ..utils.types import RelaxResult, copy_relax_result
from ..utils.core_utils import poscar_text_from_pymatgen
from ..utils.xrd_utils import r_factor_peak_pairing, wasserstein_dist

# =============================================================================
# Constants
# =============================================================================
DEFAULT_ENERGY_DELTA = 0.1
DEFAULT_R_FACTOR_DELTA = 0.05
DEFAULT_WASSERSTEIN_DELTA = 0.5

DEFAULT_DEDUP_ENERGY = 0.005
DEFAULT_DEDUP_R_FACTOR = 0.01
DEFAULT_DEDUP_WASSERSTEIN = 0.01
FAILED_METRIC_PENALTY = 99.9

DEFAULT_KEEP_MAX_RF_RATIO = 0.4
DEFAULT_KEEP_MAX_ENERGY_RATIO = 0.4
DEFAULT_KEEP_MAX_WASSERSTEIN_RATIO = 0.2

# Structure output file names
POSCAR_FILENAME = 'POSCARs'
CONTCAR_FILENAME = 'CONTCARs'
NO_SYMMETRY_CONTCAR_FILENAME = 'CONTCARs_no_sym'


def _copy_carried_candidate(candidate: RelaxResult) -> RelaxResult:
    """Copy a carried-over candidate and reset per-generation timing fields."""
    carried = copy_relax_result(candidate)
    carried.t_rel = 0.0
    carried.t_gen = 0.0
    return carried


def _get_xrd_normalize(config: Dict[str, Any]) -> str | None:
    """Return the configured XRD intensity normalization mode."""
    normalize = config.get('XRD_setting', {}).get('normalize')
    if normalize is None:
        return None
    return str(normalize).lower()


def postprocess_generation(
    relax_results: List[RelaxResult],
    no_symmetry_relax_results: List[RelaxResult],
    poscar_text_list: List[str], 
    appended_meta: List[Dict[str, Any]], 
    previous_candidates: List[RelaxResult],
    gen: int, 
    env: Dict[str, Any], 
    config: Dict[str, Any], 
    two_theta_ref: List[float], 
    intensity_ref: List[float], 
    tt_tol: float, 
    two_theta_range: Tuple[float, float],
    log_file: Path,
    no_symmetry_log_file: Path | None = None,
) -> List[RelaxResult]:
    """Post-process a generation of relaxed structures.
    
    Computes XRD metrics, selects best candidates, removes duplicates,
    and writes output files.
    
    Args:
        relax_results: List of relaxation results from current generation.
        no_symmetry_relax_results: List of relaxation results from additional non-symmetry-constrained relaxation.
        poscar_text_list: POSCAR format strings for input structures.
        appended_meta: Metadata for structures from mutation/elitism.
        previous_candidates: Relaxed candidates carried over from the
            previous generation without re-relaxation.
        gen: Current generation index.
        env: Environment dict containing 'xrd_calc', 'output_path', etc.
        config: Configuration dictionary.
        two_theta_ref: Reference 2θ peak positions.
        intensity_ref: Reference peak intensities.
        tt_tol: Tolerance for 2θ peak matching.
        two_theta_range: (min, max) range for XRD calculation.
        log_file: Path to log file for writing results.
        no_symmetry_log_file: Path to the log file for the unconstrained
            re-relaxations. Required only when ``no_symmetry_relax_results``
            is non-empty.

    Returns:
        List of deduplicated best candidates for next generation.
    """

    _apply_appended_metadata(relax_results, appended_meta)

    # Compute metrics and select from the whole generation population:
    # newly relaxed trials plus carried candidates from the previous generation.
    xrd_calc = env.get('xrd_calc')
    contcar_text_list, generation_selected = compute_metrics_and_select(
        relax_results, previous_candidates, gen, xrd_calc,
        two_theta_ref, intensity_ref, tt_tol, two_theta_range, config
    )

    write_outputs(env['output_path'], poscar_text_list, contcar_text_list)

    # Score and record the unconstrained re-relaxations. Reported only: they are
    # deliberately kept out of the selection pool below so that the candidates
    # carried forward stay symmetry-constrained.
    if no_symmetry_relax_results:
        if no_symmetry_log_file is None:
            raise ValueError(
                'no_symmetry_log_file is required when no_symmetry_relax_results '
                'is non-empty.'
            )
        process_no_symmetry_results(
            no_symmetry_relax_results=no_symmetry_relax_results,
            gen=gen,
            output_path=env['output_path'],
            no_symmetry_log_file=no_symmetry_log_file,
            xrd_calc=xrd_calc,
            two_theta_ref=two_theta_ref,
            intensity_ref=intensity_ref,
            tt_tol=tt_tol,
            two_theta_range=two_theta_range,
            config=config,
        )

    dedup_cfg = config.get('structure', {}).get('dedup', {})
    structure_matcher = env.get('structure_matcher') or StructureMatcher()
    candidates_next = deduplicate_candidates(generation_selected, dedup_cfg, structure_matcher)

    _mark_kept_candidates(candidates_next)

    # Log the full population considered in this generation.
    _write_generation_log(log_file, gen, relax_results, previous_candidates)

    return candidates_next


def compute_metrics_and_select(
    relax_results: List[RelaxResult], 
    previous_candidates: List[RelaxResult],
    gen: int, 
    xrd_calc: XRDCalculator, 
    two_theta_ref: List[float], 
    intensity_ref: List[float], 
    tt_tol: float, 
    two_theta_range: Tuple[float, float], 
    config: Dict[str, Any]
) -> Tuple[List[str], List[RelaxResult]]:
    """Compute trial metrics and select best candidates from the full population.
    
    Args:
        relax_results: List of newly relaxed trial results.
        previous_candidates: Already-relaxed candidates carried into this
            generation without re-relaxation.
        gen: Current generation index.
        xrd_calc: XRD calculator instance.
        two_theta_ref: Reference 2θ peak positions.
        intensity_ref: Reference peak intensities.
        tt_tol: Tolerance for peak matching.
        two_theta_range: Range for XRD calculation.
        config: Configuration dictionary.
    
    Returns:
        Tuple of (CONTCAR text list, selected candidates).
    """
    contcar_texts: List[str] = []
    normalize = _get_xrd_normalize(config)

    for pop_idx, result in enumerate(relax_results):
        contcar_texts.append(poscar_text_from_pymatgen(result.structure, gen, pop_idx))
        result.gen = gen
        result.pop = pop_idx
        _assign_result_metrics(
            result, xrd_calc, two_theta_ref, intensity_ref,
            tt_tol, two_theta_range, normalize,
        )

    selection_pool = relax_results + [
        _copy_carried_candidate(candidate) for candidate in previous_candidates
    ]
    selected = _select_candidates(selection_pool, config)
    
    return contcar_texts, selected


def _assign_result_metrics(
    result: RelaxResult,
    xrd_calc: XRDCalculator,
    two_theta_ref: List[float],
    intensity_ref: List[float],
    tt_tol: float,
    two_theta_range: Tuple[float, float],
    normalize: str | None,
) -> None:
    """Fill in a result's XRD metrics in place, penalising failed relaxations."""
    if result.status != 'converged':
        result.r_factor = FAILED_METRIC_PENALTY
        result.wasserstein = FAILED_METRIC_PENALTY
        result.energy = FAILED_METRIC_PENALTY
        return

    r_factor, wasserstein = analyze_relaxed_structure(
        result.structure, two_theta_ref, intensity_ref,
        xrd_calc, tt_tol, two_theta_range, normalize
    )
    result.r_factor = r_factor
    result.wasserstein = wasserstein


def process_no_symmetry_results(
    no_symmetry_relax_results: List[RelaxResult],
    gen: int,
    output_path: Path,
    no_symmetry_log_file: Path,
    xrd_calc: XRDCalculator,
    two_theta_ref: List[float],
    intensity_ref: List[float],
    tt_tol: float,
    two_theta_range: Tuple[float, float],
    config: Dict[str, Any],
) -> List[str]:
    """Score and record the unconstrained re-relaxations of a generation.

    These results are reported only. They get their own CONTCAR file and log so
    they can be compared against their symmetry-constrained counterparts, but
    they never enter the candidate pool.

    Args:
        no_symmetry_relax_results: Unconstrained results, each carrying the
            population index of the trial it was derived from.
        gen: Current generation index.
        output_path: Directory for output files.
        no_symmetry_log_file: Path to the non-symmetry log file.
        xrd_calc: XRD calculator instance.
        two_theta_ref: Reference 2θ peak positions.
        intensity_ref: Reference peak intensities.
        tt_tol: Tolerance for peak matching.
        two_theta_range: Range for XRD calculation.
        config: Configuration dictionary.

    Returns:
        The CONTCAR texts written for these results.
    """
    if not no_symmetry_relax_results:
        return []

    normalize = _get_xrd_normalize(config)
    contcar_texts: List[str] = []

    for result in no_symmetry_relax_results:
        result.gen = gen
        # pop mirrors the symmetry-constrained trial, so the two files line up.
        pop_idx = result.pop if result.pop is not None else 0
        contcar_texts.append(poscar_text_from_pymatgen(result.structure, gen, pop_idx))
        _assign_result_metrics(
            result, xrd_calc, two_theta_ref, intensity_ref,
            tt_tol, two_theta_range, normalize,
        )

    write_no_symmetry_outputs(output_path, contcar_texts)

    with open(no_symmetry_log_file, 'a') as stream:
        for result in no_symmetry_relax_results:
            _write_log_row(stream, gen, result.pop, result)

    return contcar_texts


def analyze_relaxed_structure(
    structure: Structure, 
    two_theta_ref: List[float], 
    intensity_ref: List[float], 
    xrd_calc: XRDCalculator, 
    tt_tol: float, 
    two_theta_range: Tuple[float, float],
    normalize: str | None = None,
) -> Tuple[float, float]:
    """Calculate XRD metrics for a relaxed structure.
    
    Args:
        structure: Relaxed pymatgen Structure.
        two_theta_ref: Reference 2θ peak positions.
        intensity_ref: Reference peak intensities.
        xrd_calc: XRD calculator instance.
        tt_tol: Tolerance for peak matching.
        two_theta_range: Range for XRD calculation.
        normalize: Intensity normalization mode for both reference and
            calculated patterns. One of None, "max", "sum", or "scale".
    
    Returns:
        Tuple of (R-factor, Wasserstein distance).
    """
    pattern = xrd_calc.get_pattern(structure, two_theta_range=two_theta_range)
    
    r_factor = r_factor_peak_pairing(
        two_theta_ref, intensity_ref,
        pattern.x, pattern.y,
        tol=tt_tol,
        normalize=normalize,
        two_theta_range=two_theta_range
    )
    
    wasserstein = wasserstein_dist(
        two_theta_ref, intensity_ref,
        pattern.x, pattern.y,
        two_theta_range=two_theta_range,
    )
    
    return float(r_factor), float(wasserstein)


def deduplicate_candidates(
    candidates: List[RelaxResult], 
    dedup_cfg: Dict[str, float], 
    structure_matcher: StructureMatcher
) -> List[RelaxResult]:
    """Remove duplicate structures from candidate list.
    
    Structures are considered duplicates if their metrics are within
    tolerance thresholds AND they match structurally.
    
    Args:
        candidates: List of candidate structures to deduplicate.
        dedup_cfg: Configuration dict with 'energy', 'r_factor', and
            'wasserstein' tolerances. Legacy 'wd' is also accepted.
        structure_matcher: StructureMatcher instance for structural comparison.
    
    Returns:
        List of unique candidates sorted by energy.
    """
    energy_tol = float(dedup_cfg.get('energy', DEFAULT_DEDUP_ENERGY))
    r_factor_tol = float(dedup_cfg.get('r_factor', DEFAULT_DEDUP_R_FACTOR))
    wasserstein_tol = float(
        dedup_cfg.get(
            'wasserstein',
            dedup_cfg.get('wd', DEFAULT_DEDUP_WASSERSTEIN),
        )
    )

    unique: List[RelaxResult] = []

    for candidate in sorted(candidates, key=_candidate_sort_key_RF):
        is_duplicate = _check_duplicate(
            candidate, unique, 
            energy_tol, r_factor_tol, wasserstein_tol, 
            structure_matcher
        )
        if not is_duplicate:
            unique.append(candidate)

    unique = _apply_keep_max_mix(
        unique, dedup_cfg.get('keep_max'), dedup_cfg.get('mix_ratios')
    )

    return sorted(
        unique, 
        key=lambda r: r.energy if r.energy is not None else float('inf')
    )


def write_outputs(
    output_path: Path, 
    poscar_texts: List[str], 
    contcar_texts: List[str]
) -> None:
    """Write POSCAR and CONTCAR files to output directory.
    
    Args:
        output_path: Directory path for output files.
        poscar_texts: List of POSCAR format strings.
        contcar_texts: List of CONTCAR format strings.
    """
    output_dir = Path(output_path)
    
    with open(output_dir / POSCAR_FILENAME, 'a') as poscar_file:
        for text in poscar_texts:
            poscar_file.write(text)
    
    with open(output_dir / CONTCAR_FILENAME, 'a') as contcar_file:
        for text in contcar_texts:
            contcar_file.write(text)


def write_no_symmetry_outputs(
    output_path: Path,
    contcar_texts: List[str],
) -> None:
    """Append the unconstrained relaxation structures to their own CONTCAR file.

    Args:
        output_path: Directory path for output files.
        contcar_texts: List of CONTCAR format strings.
    """
    output_dir = Path(output_path)

    with open(output_dir / NO_SYMMETRY_CONTCAR_FILENAME, 'a') as contcar_file:
        for text in contcar_texts:
            contcar_file.write(text)


# =============================================================================
# Helper Functions
# =============================================================================
def _apply_appended_metadata(
    relax_results: List[RelaxResult],
    appended_meta: List[Dict[str, Any]],
) -> None:
    """Attach appended metadata to the tail of the relaxation results list."""
    if not appended_meta:
        return

    start_idx = len(relax_results) - len(appended_meta)
    for idx, meta in enumerate(appended_meta):
        result = relax_results[start_idx + idx]
        result.parent = meta.get('parent', 'none')
        result.origin_type = meta.get('origin_type', 'kept_best')


def _mark_kept_candidates(candidates: List[RelaxResult]) -> None:
    """Mark next-generation candidates as kept-best structures."""
    for result in candidates:
        result.origin_type = 'kept_best'
        result.parent = f"{result.gen}_{result.pop}"
        result.t_gen = 0.0
        result.t_rel = 0.0


def _apply_keep_max_mix(
    candidates: List[RelaxResult],
    keep_max: Any,
    mix_ratios: Dict[str, Any] | None = None,
) -> List[RelaxResult]:
    """Apply mixed metric truncation when keep_max is configured."""
    if keep_max is None:
        return candidates

    try:
        keep_max = int(keep_max)
    except (TypeError, ValueError):
        return candidates

    if keep_max <= 0 or len(candidates) <= keep_max:
        return candidates

    mix_ratios = mix_ratios or {}
    rf_ratio = float(mix_ratios.get('r_factor', DEFAULT_KEEP_MAX_RF_RATIO))
    energy_ratio = float(mix_ratios.get('energy', DEFAULT_KEEP_MAX_ENERGY_RATIO))
    wd_ratio = float(mix_ratios.get('wasserstein', DEFAULT_KEEP_MAX_WASSERSTEIN_RATIO))

    num_RF = int(rf_ratio * keep_max)
    num_energy = int(energy_ratio * keep_max)
    num_WD = int(wd_ratio * keep_max)

    # Ensure the total number of selected candidates equals keep_max.
    residual = keep_max - (num_RF + num_energy + num_WD)
    if (num_RF >= num_energy) and (num_RF >= num_WD):
        num_RF += residual
    elif (num_energy >= num_RF) and (num_energy >= num_WD):
        num_energy += residual
    else:
        num_WD += residual

    rf_sorted = sorted(candidates, key=_candidate_sort_key_RF)
    unique_sort_by_RF = rf_sorted[:num_RF]
    remaining = rf_sorted[num_RF:]

    e_sorted = sorted(remaining, key=_candidate_sort_key_energy)
    unique_sort_by_energy = e_sorted[:num_energy]
    remaining = e_sorted[num_energy:]

    wd_sorted = sorted(remaining, key=_candidate_sort_key_WD)
    unique_sort_by_WD = wd_sorted[:num_WD]

    return unique_sort_by_RF + unique_sort_by_energy + unique_sort_by_WD


def _write_log_row(
    stream,
    gen: int,
    pop: int,
    result: RelaxResult,
) -> None:
    """Write one generation log row."""
    origin_type = getattr(result, 'origin_type', 'random')
    parent = getattr(result, 'parent', 'none')
    t_gen = result.t_gen if result.t_gen is not None else 0.0

    stream.write(
        f"{gen:>4} {pop:>4} {origin_type:>15} {parent:>8} "
        f"{result.energy:>8.4f} {result.r_factor:>8.4f} {result.wasserstein:8.4f} "
        f"{t_gen:>6.2f} {result.t_rel:>6.2f} {result.status}\n"
    )


def _write_generation_log(
    log_file: Path,
    gen: int,
    relax_results: List[RelaxResult],
    carried_candidates: List[RelaxResult],
) -> None:
    """Write newly relaxed trials followed by carried candidates."""
    with open(log_file, 'a') as f:
        for result in relax_results:
            _write_log_row(f, gen, result.pop, result)

        pop_offset = len(relax_results)
        for pool_idx, result in enumerate(carried_candidates):
            carried = _copy_carried_candidate(result)
            _write_log_row(f, gen, pop_offset + pool_idx, carried)


def _select_candidates(
    relax_results: List[RelaxResult], 
    config: Dict[str, Any]
) -> List[RelaxResult]:
    """Select best candidates based on metric thresholds."""
    str_cfg = config.get('structure', {}) if config else {}
    metrics_cfg = str_cfg.get('best_metric', {})

    # Parse thresholds
    energy_enabled, energy_delta = _parse_metric_threshold(
        metrics_cfg, 'energy', ['Energy', 'energy'], True, DEFAULT_ENERGY_DELTA
    )
    r_enabled, r_delta = _parse_metric_threshold(
        metrics_cfg, 'r_factor', ['R', 'r_factor', 'r'], True, DEFAULT_R_FACTOR_DELTA
    )
    wd_enabled, wd_delta = _parse_metric_threshold(
        metrics_cfg, 'wasserstein', ['WD', 'wd', 'wasserstein'], True, DEFAULT_WASSERSTEIN_DELTA
    )

    # Calculate minimum values
    energies = [r.energy for r in relax_results if r.energy is not None]
    r_factors = [r.r_factor for r in relax_results if r.r_factor is not None]
    wassersteins = [r.wasserstein for r in relax_results if r.wasserstein is not None]

    min_energy = min(energies) if energies else float('inf')
    min_r_factor = min(r_factors) if r_factors else float('inf')
    min_wasserstein = min(wassersteins) if wassersteins else float('inf')

    # Select candidates within threshold
    selected = [
        copy_relax_result(r) for r in relax_results 
        if _is_within_threshold(
            r, energy_enabled, energy_delta, min_energy,
            r_enabled, r_delta, min_r_factor,
            wd_enabled, wd_delta, min_wasserstein
        )
    ]

    return selected


def _parse_metric_threshold(
    cfg: Dict[str, Any], 
    key: str, 
    legacy_keys: List[str], 
    default_enabled: bool, 
    default_delta: float
) -> Tuple[bool, float]:
    """Parse metric threshold from config with legacy key support."""
    if isinstance(cfg.get(key), dict):
        return (
            cfg[key].get('enabled', default_enabled), 
            float(cfg[key].get('delta', default_delta))
        )
    
    for legacy in legacy_keys:
        if legacy in cfg:
            try:
                return True, float(cfg[legacy])
            except (ValueError, TypeError):
                return True, default_delta
    
    return default_enabled, default_delta


def _is_within_threshold(
    result: RelaxResult,
    energy_enabled: bool, energy_delta: float, min_energy: float,
    r_enabled: bool, r_delta: float, min_r_factor: float,
    wd_enabled: bool, wd_delta: float, min_wasserstein: float
) -> bool:
    """Check if a result is within selection thresholds."""
    if energy_enabled and result.energy is not None:
        if result.energy <= min_energy + energy_delta:
            return True
    if r_enabled and result.r_factor is not None:
        if result.r_factor <= min_r_factor + r_delta:
            return True
    if wd_enabled and result.wasserstein is not None:
        if result.wasserstein <= min_wasserstein + wd_delta:
            return True
    return False


def _candidate_sort_key_RF(result: RelaxResult) -> Tuple[float, float, float]:
    """Sort key for candidate selection (R-factor, Wasserstein, Energy)."""
    return (
        result.r_factor if result.r_factor is not None else float('inf'),
        result.wasserstein if result.wasserstein is not None else float('inf'),
        result.energy if result.energy is not None else float('inf'),
    )


def _candidate_sort_key_energy(result: RelaxResult) -> Tuple[float, float, float]:
    """Sort key for candidate selection (Energy, R-factor, Wasserstein)."""
    return (
        result.energy if result.energy is not None else float('inf'),
        result.r_factor if result.r_factor is not None else float('inf'),
        result.wasserstein if result.wasserstein is not None else float('inf'),
    )


def _candidate_sort_key_WD(result: RelaxResult) -> Tuple[float, float, float]:
    """Sort key for candidate selection (Wasserstein, R-factor, Energy)."""
    return (
        result.wasserstein if result.wasserstein is not None else float('inf'),
        result.r_factor if result.r_factor is not None else float('inf'),
        result.energy if result.energy is not None else float('inf'),
    )


def _check_duplicate(
    candidate: RelaxResult,
    unique_list: List[RelaxResult],
    energy_tol: float,
    r_factor_tol: float,
    wasserstein_tol: float,
    structure_matcher: StructureMatcher
) -> bool:
    """Check if candidate is a duplicate of any structure in unique_list."""
    for existing in unique_list:
        if (
            candidate.energy is None or existing.energy is None or
            candidate.r_factor is None or existing.r_factor is None or
            candidate.wasserstein is None or existing.wasserstein is None
        ):
            continue

        metrics_similar = (
            abs(candidate.energy - existing.energy) <= energy_tol and
            abs(candidate.r_factor - existing.r_factor) <= r_factor_tol and
            abs(candidate.wasserstein - existing.wasserstein) <= wasserstein_tol
        )
        
        if metrics_similar:
            try:
                if structure_matcher.fit(candidate.structure, existing.structure, symmetric=True):
                    candidate.parent = f"{existing.gen}_{existing.pop}"
                    return True
            except Exception:
                pass
    
    return False
