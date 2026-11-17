"""Structure generation module for XCAPE.

This module provides functions for generating and mutating crystal structures
using PyXtal for random structure generation and evolutionary mutations.
"""
import random
import time
from typing import Any, List, Dict, Tuple, Callable
import logging
import math

from pyxtal import pyxtal
from pyxtal.symmetry import Group
from pyxtal.lattice import Lattice
from pyxtal.tolerance import Tol_matrix

from pymatgen.core import Composition, Structure

from .mutation import Mutator

from ..utils.types import RelaxResult
from ..utils.core_utils import poscar_text_from_pymatgen, get_ltype, hm_symbol_to_hall_number

DEFAULT_RANDOM_GENERATION_MAX_ATTEMPTS = 10


def random_generation_pyxtal(
    ref_info: Dict[str, Any], 
    num_pop: int, 
    gen: int, 
    tol_matrix: Tol_matrix
) -> Tuple[List[Structure], List[str], List[float]]:
    """Generate random crystal structures using PyXtal.

    Args:
        ref_info: Reference info dict containing crystallographic data
            (chemical_formula, lattice_parameters, z, spg_num, etc.).
        num_pop: Number of structures to generate.
        gen: Generation index (used for POSCAR comments).
        tol_matrix: Tolerance matrix for interatomic distances.

    Returns:
        Tuple containing:
            - structure_list: List of generated pymatgen Structure objects.
            - poscar_list: List of POSCAR format strings.
            - time_list: List of generation times in seconds.
    """
    pretty_formula = ref_info['chemical_formula']
    lat = [round(float(x), 5) for x in ref_info['lattice_parameters']]
    ltype = get_ltype(lat)
    Z = round(ref_info['z'])
    spg_number = int(ref_info['spg_num'])
    spg_symbol = ref_info['spg_symbol']

    comp_dict = Composition(pretty_formula).get_el_amt_dict()
    comp_dict = {k: round(v * Z) for k, v in comp_dict.items()}

    # Validate space group number against symbol
    spg = Group(spg_number)
    if spg.symbol != spg_symbol:
        if spg.symbol != spg_symbol.replace("_", ""):
            hm_number = hm_symbol_to_hall_number(spg_symbol, spg_number)
            spg = Group(hm_number, use_hall=True)
            logging.warning(
                f"Provided space group symbol '{spg_symbol}' does not match conventional symbol of space group number {spg_number}. "
                f"Using HM number {hm_number} based on given symbol ({spg_symbol})."
            )

    lattice = Lattice.from_para(*lat, ltype=ltype)

    structure_list: List[Structure] = []
    poscar_list: List[str] = []
    time_list: List[float] = []
    
    for idx in range(num_pop):
        start_time = time.time()
        pmg_struct = _generate_valid_random_structure(
            spg=spg,
            lattice=lattice,
            species=list(comp_dict.keys()),
            num_ions=list(comp_dict.values()),
            tol_matrix=tol_matrix,
        )
        _append_structure_result(
            pmg_struct,
            structure_list,
            poscar_list,
            time_list,
            gen,
            start_time,
            pop_idx=idx,
        )

    return structure_list, poscar_list, time_list


def _generate_valid_random_structure(
    spg: Group,
    lattice: Lattice,
    species: List[str],
    num_ions: List[int],
    tol_matrix: Tol_matrix,
    max_attempts: int = DEFAULT_RANDOM_GENERATION_MAX_ATTEMPTS,
) -> Structure:
    """Generate a random structure from the target PyXtal spacegroup."""
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        crystal = pyxtal()
        try:
            crystal.from_random(
                dim=3,
                group=spg,
                lattice=lattice,
                species=species,
                numIons=num_ions,
                tm=tol_matrix,
                max_count=100,
            )
            return crystal.to_pymatgen()
        except Exception as exc:
            last_error = exc
            logging.warning(
                "Random structure generation failed on attempt %s/%s: %s",
                attempt,
                max_attempts,
                exc,
            )

    raise RuntimeError(
        "Failed to generate a random structure with PyXtal after "
        f"{max_attempts} attempts."
    ) from last_error


def perform_mutation_and_append(
    candidates: List[RelaxResult],
    num_mut: int,
    structure_list: List[Structure],
    poscar_text_list: List[str],
    t_gen_list: List[float],
    gen: int,
    tol_matrix: Tol_matrix,
    config: Dict[str, Any],
    spg_num: int | None = None,
) -> List[Dict[str, Any]]:
    """Perform mutations on candidate structures and append results.
    
    Selects parent structures using Boltzmann-weighted sampling based on
    a configurable metric, then applies random mutations.
    
    Args:
        candidates: List of relaxation results to select parents from.
        num_mut: Number of mutations to perform.
        structure_list: List to append mutated structures to (modified in-place).
        poscar_text_list: List to append POSCAR strings to (modified in-place).
        t_gen_list: List to append generation times to (modified in-place).
        gen: Current generation index.
        tol_matrix: Tolerance matrix for mutation distance constraints.
        config: Configuration dictionary.
    
    Returns:
        List of metadata dicts for each appended structure, containing
        'parent', 'origin_type', and 'mutate_name' keys.
    """
    appended_meta = []
    if num_mut <= 0 or len(candidates) == 0:
        return appended_meta

    mutation_settings = _get_mutation_settings(config, gen, spg_num)
    mutator = Mutator(
        **mutation_settings['mutation_ratios'],
        tolerance_matrix=tol_matrix,
    )
    metric_fn = _create_metric_function(mutation_settings['parent_metric'])
    weights = build_boltzmann_weights(
        candidates,
        metric_fn,
        mutation_settings['temperature'],
    )

    for _ in range(int(num_mut)):
        start_time = time.time()
        parent_idx = _select_parent_index(
            candidates,
            weights,
            mutation_settings['weighted_parent_ratio'],
        )
        parent_result = candidates[parent_idx]
        parent_struct = parent_result.structure
        xtal_struct = _prepare_structure_for_mutation(
            parent_struct,
            mutation_settings['init_spg_num'],
        )
        if xtal_struct is None:
            logging.warning(
                f"Failed to convert structure to initial spacegroup "
                f"{mutation_settings['init_spg_num']}. Using parent structure without mutation."
            )
            _append_structure_result(
                parent_struct, structure_list, poscar_text_list,
                t_gen_list, gen, start_time
            )
            appended_meta.append(_build_meta(parent_result, 'spg_mismatch'))
            continue

        mutated_pmg, mutate_name = _perform_single_mutation(
            xtal_struct,
            mutator,
            parent_struct,
            mutation_settings['max_attempts'],
        )
        
        _append_structure_result(
            mutated_pmg, structure_list, poscar_text_list, 
            t_gen_list, gen, start_time
        )
        appended_meta.append(_build_meta(parent_result, mutate_name))

    return appended_meta


def _get_mutation_settings(config: Dict[str, Any], gen: int, spg_num: int | None = None) -> Dict[str, Any]:
    """Extract mutation-related settings with defaults applied."""
    mutation_config = config.get('structure', {}).get('mutation', {})
    selection_method = mutation_config.get('selection_method', {})

    temperature = float(selection_method.get('temperature', 1.0))
    temperature_decay = float(selection_method.get('temperature_decay', 0.9))
    min_temperature = float(selection_method.get('min_temperature', 0.5))

    return {
        'mutation_ratios': mutation_config.get('mutation_ratios', {}),
        'max_attempts': mutation_config.get('max_mutation_attempts', 100),
        'parent_metric': selection_method.get('parent_metric', 'energy'),
        'weighted_parent_ratio': float(selection_method.get('weighted_parent_ratio', 0.8)),
        'temperature': max(
            temperature * (temperature_decay ** (gen - 1)),
            min_temperature,
        ),
        'init_spg_num': spg_num,
    }


def _select_parent_index(
    candidates: List[RelaxResult],
    weights: List[float],
    weighted_parent_ratio: float,
) -> int:
    """Select a parent index with weighted sampling and random fallback."""
    try:
        if random.random() <= weighted_parent_ratio:
            return random.choices(range(len(candidates)), weights=weights, k=1)[0]
        return random.randrange(len(candidates))
    except Exception as exc:
        logging.warning(f"Error occurred while selecting parent: {exc}")
        return random.randrange(len(candidates))


def _prepare_structure_for_mutation(
    parent_struct: Structure,
    init_spg_num: Any,
) -> pyxtal | None:
    """Convert a parent structure to PyXtal and align spacegroup if needed."""
    xtal_struct = pyxtal()
    try:
        xtal_struct.from_seed(parent_struct)
    except Exception as exc:
        logging.warning("Failed to convert parent structure to PyXtal seed: %s", exc)
        return None

    if init_spg_num is None or xtal_struct.group.number == init_spg_num:
        return xtal_struct

    resym_xtal_struct = _resymmetrize_to_target_group(xtal_struct, init_spg_num)
    if resym_xtal_struct is not None:
        return resym_xtal_struct

    logging.warning(
        f"Parent structure spacegroup {xtal_struct.group.number} is not same as "
        f"the initial spacegroup {init_spg_num}. Try changing into {init_spg_num}."
    )
    return _convert_to_target_group(xtal_struct, init_spg_num)


def _resymmetrize_to_target_group(
    xtal_struct: pyxtal,
    target_spg_num: int,
) -> pyxtal | None:
    """Try to resymmetrize a structure into the target spacegroup."""
    for tol in [1.0, 2e-1, 1e-1, 1e-2, 1e-4, 1e-5]:
        try:
            resym_xtal_struct = xtal_struct.resymmetrize(tol=tol)
        except Exception as exc:
            logging.warning(
                "Resymmetrization failed for target spacegroup %s with tol %s: %s",
                target_spg_num,
                tol,
                exc,
            )
            continue

        if resym_xtal_struct is None:
            continue

        if resym_xtal_struct.group.number == target_spg_num:
            logging.info(
                f"Successfully resymmetrized structure to spacegroup "
                f"{target_spg_num} with tol {tol}."
            )
            return resym_xtal_struct
    return None


def _convert_to_target_group(
    xtal_struct: pyxtal,
    target_spg_num: int,
) -> pyxtal | None:
    """Try subgroup and supergroup transformations to reach the target spacegroup."""
    try:
        sub_xtal_struct = xtal_struct.subgroup_once(H=target_spg_num, mut_lat=False)
        if sub_xtal_struct is not None and sub_xtal_struct.group.number == target_spg_num:
            logging.info(
                f"Successfully converted structure to spacegroup {target_spg_num} "
                f"using subgroup transformation."
            )
            return sub_xtal_struct
    except Exception as exc:
        logging.warning(
            f"Subgroup transformation failed with error: {exc}. "
            f"Trying supergroup transformation."
        )

    try:
        super_xtal_structs, _ = xtal_struct.supergroup(G=target_spg_num)
        super_xtal_struct = super_xtal_structs[0] if super_xtal_structs else None
        if super_xtal_struct is not None and super_xtal_struct.group.number == target_spg_num:
            logging.info(
                f"Successfully converted structure to spacegroup {target_spg_num} "
                f"using supergroup transformation."
            )
            return super_xtal_struct
    except Exception as exc:
        logging.warning(f"Supergroup transformation failed with error: {exc}")

    return None


def _create_metric_function(metric_name: str) -> Callable[[RelaxResult], float]:
    """Create a metric function for parent selection.
    
    Args:
        metric_name: Name of metric ('energy', 'r_factor', 'wasserstein').
    
    Returns:
        Function that extracts the metric value from a RelaxResult.
    """
    def metric_fn(result: RelaxResult) -> float:
        if metric_name == 'energy':
            return result.energy if result.energy is not None else float('inf')
        if metric_name in ('r', 'r_factor'):
            return result.r_factor if result.r_factor is not None else float('inf')
        if metric_name in ('wd', 'wasserstein'):
            return result.wasserstein if result.wasserstein is not None else float('inf')
        return result.energy if result.energy is not None else float('inf')
    return metric_fn


def _perform_single_mutation(
    xtal_struct: pyxtal, 
    mutator: Mutator, 
    fallback_struct: Structure,
    max_attempts: int,
) -> Tuple[Structure, str]:
    """Attempt mutation on a single structure.
    
    Args:
        xtal_struct: PyXtal structure to mutate.
        mutator: Mutator instance.
        fallback_struct: Structure to return if mutation fails.
        max_attempts: Maximum number of mutation attempts.
    
    Returns:
        Tuple of (mutated structure, mutation name).
    """
    for _ in range(max_attempts):
        mutated_xtal, success, mutate_name = mutator.mutate(xtal_struct)
        if success:
            return mutated_xtal.to_pymatgen(), mutate_name
    
    logging.warning("Max mutation attempts reached, using parent structure.")
    return fallback_struct, 'max_attempts_reached'


def _append_structure_result(
    structure: Structure,
    structure_list: List[Structure],
    poscar_text_list: List[str],
    t_gen_list: List[float],
    gen: int,
    start_time: float,
    pop_idx: int | None = None,
    elapsed_override: float | None = None,
) -> None:
    """Append a structure and its metadata to the result lists."""
    structure_list.append(structure)
    poscar_text_list.append(
        poscar_text_from_pymatgen(
            structure,
            gen,
            len(poscar_text_list) if pop_idx is None else pop_idx,
        )
    )
    if elapsed_override is not None:
        t_gen_list.append(elapsed_override)
    else:
        t_gen_list.append(time.time() - start_time)


def _build_meta(parent_result: RelaxResult, origin_type: str) -> Dict[str, Any]:
    """Build metadata dictionary for an appended structure."""
    parent_str = _resolve_parent_id(parent_result)
    return {
        'parent': parent_str,
        'origin_type': origin_type,
        'mutate_name': origin_type,
        'orig_gen': getattr(parent_result, 'gen', None),
        'orig_pop': getattr(parent_result, 'pop', None),
    }


def _resolve_parent_id(result: RelaxResult) -> str:
    """Resolve a stable parent identifier from a RelaxResult."""
    parent_str = getattr(result, 'parent', None)
    if parent_str is not None:
        return parent_str
    if result.gen is not None and result.pop is not None:
        return f"{result.gen}_{result.pop}"
    return 'none'


def append_previous_candidates(
    candidates: List[RelaxResult], 
    structure_list: List[Structure], 
    poscar_text_list: List[str], 
    t_gen_list: List[float], 
    gen: int
) -> List[Dict[str, Any]]:
    """Append previous best candidates to the current generation.
    
    Used to carry over elite structures from previous generations.
    
    Args:
        candidates: List of previous best RelaxResult objects.
        structure_list: List to append structures to (modified in-place).
        poscar_text_list: List to append POSCAR strings to (modified in-place).
        t_gen_list: List to append times to (modified in-place).
        gen: Current generation index.
    
    Returns:
        List of metadata dicts for each appended structure.
    """
    appended_meta = []
    
    for result in candidates:
        appended_meta.append(_build_meta(result, 'kept_best'))

        _append_structure_result(
            result.structure,
            structure_list,
            poscar_text_list,
            t_gen_list,
            gen,
            start_time=0.0,
            elapsed_override=0.0,
        )
    
    return appended_meta


def build_boltzmann_weights(
    candidates: List[RelaxResult], 
    metric_fn: Callable[[RelaxResult], float], 
    temperature: float
) -> List[float]:
    """Calculate Boltzmann-weighted selection probabilities.
    
    Higher weights are assigned to candidates with lower metric values.
    
    Args:
        candidates: List of candidate structures.
        metric_fn: Function to extract metric value from RelaxResult.
        temperature: Boltzmann temperature (higher = more uniform selection).
    
    Returns:
        List of normalized weights summing to 1.0.
    """
    metrics = [metric_fn(candidate) for candidate in candidates]
    finite_metrics = [m for m in metrics if m != float('inf')]
    
    # Handle edge case: no valid metrics
    if len(finite_metrics) == 0:
        return [1.0 / len(metrics)] * len(metrics)
    
    metric_min = min(finite_metrics)
    
    # Handle zero temperature: select only best candidates
    if temperature <= 0:
        best_indices = [i for i, m in enumerate(metrics) if m == metric_min]
        weights = [0.0] * len(metrics)
        for idx in best_indices:
            weights[idx] = 1.0 / len(best_indices)
        return weights
    
    # Calculate Boltzmann weights
    exp_values = [
        0.0 if m == float('inf')
        else math.exp(-(m - metric_min) / temperature)
        for m in metrics
    ]
    total = sum(exp_values)
    
    if total == 0:
        return [1.0 / len(metrics)] * len(metrics)
    
    return [exp_val / total for exp_val in exp_values]
