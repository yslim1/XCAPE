"""Main entry point for the XCAPE workflow."""

import argparse
import logging
from pathlib import Path
from typing import Any, List, Tuple

import psutil
import torch

from .modules.generation import (
    perform_mutation_and_append
)
from .modules.relaxation import serial_relax, batch_relax
from .modules.postprocess import (
    NO_SYMMETRY_CONTCAR_FILENAME,
    postprocess_generation,
)
from .mlips.sevennet import load_sevennet_calc
from .reference import prepare_reference, ensure_distance_constraint
from .runtime import (
    setup_environment,
    setup_logging,
    initialize_generation_log,
    LOG_FILENAME,
    MEMORY_LOG_FILENAME,
    NO_SYMMETRY_LOG_FILENAME,
)
from .validation import (
    DEFAULT_ADD_NO_SYMMETRY_RELAX_RATIO,
    DEFAULT_EARLY_STOPPING_THRESHOLD,
    validate_config,
)
from .utils.types import Environment, ReferenceInfo, RelaxResult
from .utils.core_utils import load_config, dump_yaml, set_global_seeds

logger = logging.getLogger(__name__)
logging.getLogger("torch_sim.autobatching").setLevel(logging.WARNING)


def early_stopping(candidates: List[RelaxResult], threshold: float) -> bool:
    """Check for early stopping condition based on candidate convergence."""
    return any(
        candidate.r_factor is not None and candidate.r_factor < threshold
        for candidate in candidates
    )


def _early_stopping_reason(
    candidates: List[RelaxResult],
    no_symmetry_results: List[RelaxResult],
    threshold: float,
) -> str | None:
    """Describe why the run should stop early, or None to keep going."""
    if early_stopping(candidates, threshold):
        return 'convergence criteria met'
    if early_stopping(no_symmetry_results, threshold):
        return (
            'convergence criteria met by a non-symmetry-constrained structure '
            f'(recorded in {NO_SYMMETRY_CONTCAR_FILENAME} and '
            f'{NO_SYMMETRY_LOG_FILENAME}, not in the candidate pool)'
        )
    return None


def rss_mb() -> float:
    """Get current RSS memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024**2


def _get_early_stopping_threshold(config: dict[str, Any]) -> float:
    """Return the configured early stopping threshold."""
    threshold = config.get('structure', {}).get(
        'early_stopping_threshold',
        DEFAULT_EARLY_STOPPING_THRESHOLD,
    )
    return float(threshold)


def _resolve_two_theta_range(
    two_theta_ref: List[float],
    xrd_cfg: dict[str, Any],
) -> Tuple[float, float]:
    """Return the internal comparison range, expanded by two_theta_tol."""
    tt_tol = float(xrd_cfg.get('two_theta_tol', 0.0))
    config_two_theta_range = xrd_cfg.get('two_theta_range')
    if config_two_theta_range is not None:
        return (
            float(config_two_theta_range[0]) - tt_tol,
            float(config_two_theta_range[1]) + tt_tol,
        )
    return (min(two_theta_ref) - tt_tol, max(two_theta_ref) + tt_tol)


def _build_distance_constraint_tol_matrix(
    config: dict[str, Any],
    env: Environment,
) -> Any:
    """Build distance constraints using a temporary sequential MLIP backend."""
    if config['optimizer']['type'] == 'sequential':
        return ensure_distance_constraint(config, env['mlip'])

    distance_mlip = load_sevennet_calc(config, optimizer_type='sequential')
    try:
        return ensure_distance_constraint(config, distance_mlip)
    finally:
        del distance_mlip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_generation_population(
    env: Environment,
    ref_info: ReferenceInfo,
    candidates: List[RelaxResult],
    gen: int,
    tol_matrix,
    config: dict[str, Any],
) -> Tuple[List[Any], List[str], List[float], List[dict[str, Any]]]:
    """Build the population to be newly relaxed for one generation."""
    n_populations = config['structure']['population']
    random_ratio = config['structure']['random_ratio']

    if not candidates:
        num_rand_pop = n_populations
        num_mut_pop = 0
    else:
        num_rand_pop = int(n_populations * random_ratio)
        num_mut_pop = n_populations - num_rand_pop

    structure_list, poscar_text_list, t_gen_list = env['generator'](
        ref_info,
        num_rand_pop,
        gen,
        tol_matrix,
    )

    appended_meta: List[dict[str, Any]] = []
    if candidates:
        appended_meta = perform_mutation_and_append(
            candidates=candidates,
            num_mut=num_mut_pop,
            structure_list=structure_list,
            poscar_text_list=poscar_text_list,
            t_gen_list=t_gen_list,
            gen=gen,
            tol_matrix=tol_matrix,
            config=config,
            spg_num=ref_info['spg_num'],
        )

    return structure_list, poscar_text_list, t_gen_list, appended_meta


def _run_relaxation(
    structure_list: List[Any],
    t_gen_list: List[float],
    env: Environment,
    tol_matrix,
    config: dict[str, Any],
):
    """Run relaxation with the configured optimizer backend."""
    optimizer_type = config['optimizer']['type']
    if optimizer_type == 'sequential':
        return serial_relax(
            structure_list,
            t_gen_list,
            env['mlip'],
            tol_matrix,
            config,
        )
    if optimizer_type == 'batch':
        return batch_relax(
            structure_list,
            t_gen_list,
            env['mlip'],
            tol_matrix,
            config,
        )
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def _select_no_symmetry_targets(
    relax_results: List[RelaxResult],
    config: dict[str, Any],
) -> List[int]:
    """Pick the indices of the lowest-energy converged trials of a generation."""
    ratio = float(
        config.get('structure', {}).get(
            'add_no_symmetry_relax_ratio',
            DEFAULT_ADD_NO_SYMMETRY_RELAX_RATIO,
        )
    )
    if ratio <= 0.0:
        return []

    eligible = [
        idx for idx, result in enumerate(relax_results)
        if result.status == 'converged' and result.energy is not None
    ]
    if not eligible:
        return []

    # Round down, but always keep at least one target so that a small population
    # combined with a small ratio does not silently disable the gate.
    num_targets = max(1, int(len(relax_results) * ratio))
    eligible.sort(key=lambda idx: relax_results[idx].energy)
    return eligible[:num_targets]


def _run_no_symmetry_relaxation(
    relax_results: List[RelaxResult],
    structure_list: List[Any],
    t_gen_list: List[float],
    gen: int,
    env: Environment,
    tol_matrix,
    config: dict[str, Any],
) -> List[RelaxResult]:
    """Relax the best trials of a generation again without symmetry constraints. 
    The gate is opt-in via ``structure.add_no_symmetry_relax``."""

    if not config.get('structure', {}).get('add_no_symmetry_relax', False):
        return []

    target_indices = _select_no_symmetry_targets(relax_results, config)
    if not target_indices:
        logger.info(
            "Skipping the non-symmetry-constrained relaxation for generation "
            f"{gen}: no converged candidate qualified."
        )
        return []

    logger.info(
        f"Performing additional non-symmetry-constrained relaxation for "
        f"{len(target_indices)} low-energy candidates of generation {gen}."
    )

    # Nested copy: a shallow config.copy() would share the 'optimizer' dict and
    # permanently disable fix_symmetry for every later generation.
    no_symmetry_config = {
        **config,
        'optimizer': {**config['optimizer'], 'fix_symmetry': False},
    }

    results = _run_relaxation(
        [structure_list[idx] for idx in target_indices],
        [t_gen_list[idx] for idx in target_indices],
        env,
        tol_matrix,
        no_symmetry_config,
    )

    # Keep each result addressable back to its symmetry-constrained counterpart.
    for result, source_idx in zip(results, target_indices):
        result.gen = gen
        result.pop = source_idx
        result.origin_type = 'non-sym'
        result.parent = f"{gen}_{source_idx}"

    return results


def _log_memory_usage(output_path: Path, generation: int) -> None:
    """Append the current RSS usage to the memory log."""
    with open(output_path / MEMORY_LOG_FILENAME, 'a') as stream:
        stream.write(f"[RSS] {generation}: {rss_mb():8.2f} MB\n")


def _postprocess_generation_step(
    relax_results: List[RelaxResult],
    no_symmetry_relax_results: List[RelaxResult],
    poscar_text_list: List[str],
    appended_meta: List[dict[str, Any]],
    previous_candidates: List[RelaxResult],
    gen: int,
    env: Environment,
    config: dict[str, Any],
    two_theta_ref: List[float],
    intensity_ref: List[float],
    tt_tol: float,
    two_theta_range: Tuple[float, float],
    log_file: Path,
    no_symmetry_log_file: Path,
) -> List[RelaxResult]:
    """Postprocess one generation and return the surviving candidates."""
    return postprocess_generation(
        relax_results=relax_results,
        no_symmetry_relax_results=no_symmetry_relax_results,
        poscar_text_list=poscar_text_list,
        appended_meta=appended_meta,
        previous_candidates=previous_candidates,
        gen=gen,
        env=env,
        config=config,
        two_theta_ref=two_theta_ref,
        intensity_ref=intensity_ref,
        tt_tol=tt_tol,
        two_theta_range=two_theta_range,
        log_file=log_file,
        no_symmetry_log_file=no_symmetry_log_file,
    )


def _run_generation(
    gen: int,
    env: Environment,
    ref_info: ReferenceInfo,
    candidates: List[RelaxResult],
    tol_matrix,
    config: dict[str, Any],
    two_theta_ref: List[float],
    intensity_ref: List[float],
    tt_tol: float,
    two_theta_range: Tuple[float, float],
    log_file: Path,
    no_symmetry_log_file: Path,
) -> Tuple[List[RelaxResult], List[RelaxResult]]:
    """Run one XCAPE generation.

    Returns:
        A ``(candidates, no_symmetry_results)`` pair. Only ``candidates`` is fed
        back into the next generation; ``no_symmetry_results`` is returned so the
        caller can test it for early stopping, and is empty when the gate is off.
    """
    logger.info(f"=== Generation {gen} ===")

    structure_list, poscar_text_list, t_gen_list, appended_meta = _build_generation_population(
        env=env,
        ref_info=ref_info,
        candidates=candidates,
        gen=gen,
        tol_matrix=tol_matrix,
        config=config,
    )

    relax_results = _run_relaxation(
        structure_list,
        t_gen_list,
        env,
        tol_matrix,
        config,
    )

    # Optional gate: re-relax the best trials without symmetry constraints.
    no_symmetry_relax_results = _run_no_symmetry_relaxation(
        relax_results=relax_results,
        structure_list=structure_list,
        t_gen_list=t_gen_list,
        gen=gen,
        env=env,
        tol_matrix=tol_matrix,
        config=config,
    )

    # Post-processing scores no_symmetry_relax_results in place, so the list is
    # metric-complete once this returns.
    next_candidates = _postprocess_generation_step(
        relax_results=relax_results,
        no_symmetry_relax_results=no_symmetry_relax_results,
        poscar_text_list=poscar_text_list,
        appended_meta=appended_meta,
        previous_candidates=candidates,
        gen=gen,
        env=env,
        config=config,
        two_theta_ref=two_theta_ref,
        intensity_ref=intensity_ref,
        tt_tol=tt_tol,
        two_theta_range=two_theta_range,
        log_file=log_file,
        no_symmetry_log_file=no_symmetry_log_file,
    )

    return next_candidates, no_symmetry_relax_results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the XCAPE entry point."""
    default_config = Path.cwd() / 'xcape' / 'config.yaml'

    parser = argparse.ArgumentParser(
        description='Run the XCAPE crystal structure prediction workflow.',
    )
    parser.add_argument(
        '--config',
        default=str(default_config),
        help=f'Path to the YAML configuration file (default: {default_config})',
    )
    return parser.parse_args()


def main(config_path: str) -> int:
    """Run the XCAPE crystal structure prediction algorithm.
    
    Performs iterative generation, relaxation, and selection of crystal structures
    based on powder XRD pattern matching.

    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Exit code (0 for success)
    """
    config = load_config(config_path)
    validate_config(config)
    output_path = Path(config['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path)
    # Seed before any RNG use (generation, mutation, and the MQA MD run inside
    # the distance-constraint bootstrap all draw from the global RNGs).
    seed = config.get('seed')
    if seed is not None:
        logger.info(f"Seeding global RNGs with seed={int(seed)} for reproducibility.")
    set_global_seeds(seed)
    env = setup_environment(config)
    ref_info = prepare_reference(config)
    two_theta_ref = ref_info['two_theta']
    intensity_ref = ref_info['intensity']
    if not two_theta_ref or not intensity_ref:
        raise ValueError('Reference pattern is empty. Please check the input XML/CIF source.')

    xrd_cfg = config['XRD_setting']
    tt_tol = xrd_cfg.get('two_theta_tol', 0.0)
    two_theta_range = _resolve_two_theta_range(two_theta_ref, xrd_cfg)

    tol_matrix = _build_distance_constraint_tol_matrix(config, env)
    dump_yaml(str(env['output_path'] / 'total.yaml'), config)

    log_file = env['output_path'] / LOG_FILENAME
    initialize_generation_log(log_file)

    # The non-symmetry log is only created when the optional gate is enabled.
    no_symmetry_log_file = env['output_path'] / NO_SYMMETRY_LOG_FILENAME
    if config.get('structure', {}).get('add_no_symmetry_relax', False):
        initialize_generation_log(no_symmetry_log_file)

    n_generations = config['structure']['generation']
    early_stopping_threshold = _get_early_stopping_threshold(config)

    candidates: List[RelaxResult] = []

    for gen in range(1, n_generations + 1):
        candidates, no_symmetry_results = _run_generation(
            gen=gen,
            env=env,
            ref_info=ref_info,
            candidates=candidates,
            tol_matrix=tol_matrix,
            config=config,
            two_theta_ref=two_theta_ref,
            intensity_ref=intensity_ref,
            tt_tol=tt_tol,
            two_theta_range=two_theta_range,
            log_file=log_file,
            no_symmetry_log_file=no_symmetry_log_file,
        )

        _log_memory_usage(env['output_path'], gen)

        stop_reason = _early_stopping_reason(
            candidates, no_symmetry_results, early_stopping_threshold
        )
        if stop_reason is not None:
            logger.info(f"Early stopping at generation {gen}: {stop_reason}.")
            break

    return 0


def cli() -> int:
    """Console-script entry point (installed as the ``xcape`` command)."""
    args = parse_args()
    return main(config_path=args.config)


if __name__ == '__main__':
    raise SystemExit(cli())
