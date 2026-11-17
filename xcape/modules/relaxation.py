"""Relaxation module for XCAPE.

This module provides structure relaxation functionality using ASE optimizers
with interatomic distance constraint checking.
"""
import time
import logging
from typing import Any, Dict, List
import torch

from ase import Atoms
from ase.optimize import FIRE
from ase.constraints import FixSymmetry
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ..utils.types import RelaxResult
from ..utils.torchsim_utils import (
    DEFAULT_MAX_MEMORY_PADDING,
    get_max_memory_scaler,
    optimize_with_oom_fallback,
    update_cached_raw_scaler,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================
DEFAULT_FMAX = 0.05
DEFAULT_STEPS_PER_ATOM = 100
DEFAULT_SHORT_STEPS_PER_ATOM = 1


def _validate_relaxation_inputs(
    structure_list: List[Structure],
    t_gen_list: List[float],
) -> None:
    """Validate aligned relaxation inputs before running optimizers."""
    if len(structure_list) != len(t_gen_list):
        raise ValueError(
            "structure_list and t_gen_list must have the same length: "
            f"{len(structure_list)} != {len(t_gen_list)}"
        )


def check_distance_tolerance(atoms: Atoms, tol_matrix) -> bool:
    """Check that interatomic distances obey the provided tolerance matrix.

    Args:
        atoms: ASE Atoms object to check.
        tol_matrix: Tolerance matrix providing minimum allowed distances 
            between atom types (PyXtal Tol_matrix).

    Returns:
        True if all pair distances are above their tolerance, False otherwise.
    """
    from ase.neighborlist import neighbor_list

    # Use the largest pairwise tolerance as the neighbor-list search radius.
    max_cutoff = 0.0
    chemical_symbols = list(set(atoms.get_chemical_symbols()))
    for sym1 in chemical_symbols:
        for sym2 in chemical_symbols:
            tol = tol_matrix.get_tol(sym1, sym2)
            if tol > max_cutoff:
                max_cutoff = tol

    i, j, d = neighbor_list('ijd', atoms, cutoff=max_cutoff)

    for idx in range(len(i)):
        if i[idx] < j[idx]:  # Avoid double counting.
            sym_i = atoms[i[idx]].symbol
            sym_j = atoms[j[idx]].symbol
            min_distance = tol_matrix.get_tol(
                sym_i,
                sym_j,
            )
            if d[idx] < min_distance:
                return False

    return True


def _get_optimizer_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract optimizer settings with defaults applied."""
    optimizer_cfg = config.get('optimizer', {})
    return {
        'fmax': optimizer_cfg.get('ftol', DEFAULT_FMAX),
        'steps_per_atom': optimizer_cfg.get('max_step_per_atom', DEFAULT_STEPS_PER_ATOM),
        'short_steps_per_atom': optimizer_cfg.get('max_short_step_per_atom', DEFAULT_SHORT_STEPS_PER_ATOM),
        'fix_symmetry': optimizer_cfg.get('fix_symmetry', False),
        'max_memory_padding': optimizer_cfg.get('max_memory_padding', DEFAULT_MAX_MEMORY_PADDING),
    }


def _determine_relaxation_status(
    atoms: Atoms,
    tol_matrix,
    converged: bool,
) -> str:
    """Determine the final relaxation status from geometry and convergence."""
    if not check_distance_tolerance(atoms, tol_matrix):
        return 'distance_violation_final'
    if converged:
        return 'converged'
    return 'not_converged'


def _build_relax_result(
    structure: Structure,
    energy_per_atom: float,
    start_time: float,
    t_gen: float,
    status: str,
    elapsed_time: float | None = None,
) -> RelaxResult:
    """Build a RelaxResult from normalized relaxation outputs."""
    if elapsed_time is None:
        elapsed_time = time.time() - start_time
    return RelaxResult(
        structure=structure,
        energy=energy_per_atom,
        status=status,
        t_rel=elapsed_time,
        t_gen=t_gen,
    )


def serial_relax(
    structure_list: List[Structure], 
    t_gen_list: List[float],
    calc, 
    tol_matrix, 
    config: Dict[str, Any]
) -> List[RelaxResult]:
    """Relax structures serially using ASE FIRE optimizer.
    
    Performs a two-stage relaxation: first a short optimization to check
    for distance constraint violations, then a longer full optimization.

    Args:
        structure_list: List of pymatgen Structure objects to relax.
        t_gen_list: Generation times aligned with ``structure_list``.
        calc: ASE calculator instance (e.g., SevenNet).
        tol_matrix: Tolerance matrix for minimum interatomic distances.
        config: Configuration dictionary with optimizer settings:
            - optimizer.ftol: Force tolerance (default: 0.05 eV/Å)
            - optimizer.max_step_per_atom: Max steps per atom (default: 100)
            - optimizer.max_short_step_per_atom: Initial check steps (default: 1)
            - optimizer.fix_symmetry: Whether to fix symmetry (default: False)

    Returns:
        List of RelaxResult objects containing relaxed structures,
        energies, status messages, and relaxation times.
    """
    if not structure_list:
        return []
    _validate_relaxation_inputs(structure_list, t_gen_list)
    
    adaptor = AseAtomsAdaptor()
    settings = _get_optimizer_settings(config)
    fmax = settings['fmax']
    steps_per_atom = settings['steps_per_atom']
    short_steps_per_atom = settings['short_steps_per_atom']
    fix_symmetry = settings['fix_symmetry']

    relax_results: List[RelaxResult] = []

    for idx, structure in enumerate(structure_list):
        start_time = time.time()
        
        atoms = adaptor.get_atoms(structure, msonable=False)
        atoms.calc = calc
        if fix_symmetry:
            atoms.set_constraint(FixSymmetry(atoms, symprec=1e-5))

        optimizer = FIRE(atoms, logfile=None)
        num_atoms = len(atoms)

        # Stage 1: Short optimization to check initial distance constraints
        optimizer.run(fmax=fmax, steps=short_steps_per_atom * num_atoms)
        
        if not check_distance_tolerance(atoms, tol_matrix):
            result = _build_relax_result_for_serial(
                atoms, adaptor, start_time,
                t_gen_list[idx],
                status='distance_violation_initial'
            )
            relax_results.append(result)
            continue

        # Stage 2: Full optimization
        converged = optimizer.run(fmax=fmax, steps=steps_per_atom * num_atoms)
        status = _determine_relaxation_status(atoms, tol_matrix, converged)
        
        result = _build_relax_result_for_serial(
            atoms,
            adaptor,
            start_time,
            t_gen_list[idx],
            status=status,
        )
        relax_results.append(result)

    return relax_results


def batch_relax(
    structure_list: List[Structure], 
    t_gen_list: List[float],
    calc, 
    tol_matrix, 
    config: Dict[str, Any]
) -> List[RelaxResult]:
    """Relax structures in batch using FIRE optimizer implemented in Torch-Sim project.
    
    Performs a two-stage relaxation: first a short optimization to check
    for distance constraint violations, then a longer full optimization.

    Args:
        structure_list: List of pymatgen Structure objects to relax.
        t_gen_list: Generation times aligned with ``structure_list``.
        calc: Torch-Sim compatible batched model (e.g., SevenNetModel).
        tol_matrix: Tolerance matrix for minimum interatomic distances.
        config: Configuration dictionary with optimizer settings:
            - optimizer.ftol: Force tolerance (default: 0.05 eV/Å)
            - optimizer.max_step_per_atom: Max steps per atom (default: 100)
            - optimizer.max_short_step_per_atom: Initial check steps (default: 1)
            - optimizer.fix_symmetry: Whether to fix symmetry (default: False)

    Returns:
        List of RelaxResult objects containing relaxed structures,
        energies, status messages, and wall-clock times. For batch mode,
        ``t_rel`` records the time until the structure exited stage 1 or
        completed the final optimization stage.
    """
    if not structure_list:
        return []
    _validate_relaxation_inputs(structure_list, t_gen_list)
    settings = _get_optimizer_settings(config)
    if not settings['fix_symmetry']:
        return _run_batch_relaxation(
            structure_list,
            t_gen_list,
            calc,
            tol_matrix,
            config,
            apply_fix_symmetry=False,
        )

    try:
        return _run_batch_relaxation(
            structure_list,
            t_gen_list,
            calc,
            tol_matrix,
            config,
            apply_fix_symmetry=True,
        )
    except ValueError as exc:
        logger.warning(
            "Batch FixSymmetry initialization failed with ValueError; retrying "
            "with per-structure screening so only incompatible structures fall "
            "back to unconstrained relaxation. Error: %s",
            exc,
        )

    return _batch_relax_with_symmetry_fallback(
        structure_list,
        t_gen_list,
        calc,
        tol_matrix,
        config,
    )


def _run_batch_relaxation(
    structure_list: List[Structure],
    t_gen_list: List[float],
    calc,
    tol_matrix,
    config: Dict[str, Any],
    *,
    apply_fix_symmetry: bool,
) -> List[RelaxResult]:
    """Run batch relaxation, optionally applying Torch-Sim symmetry constraints."""
    import torch_sim as ts
    from torch_sim.constraints import FixSymmetry as ts_FixSymmetry

    adaptor = AseAtomsAdaptor()
    settings = _get_optimizer_settings(config)
    fmax = settings['fmax']
    steps_per_atom = settings['steps_per_atom']
    short_steps_per_atom = settings['short_steps_per_atom']
    max_memory_padding = settings['max_memory_padding']

    relax_results: List[RelaxResult] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    force_convergence_fn = ts.generate_force_convergence_fn(
        force_tol=fmax,
        include_cell_forces=False,
    )

    num_atoms = max(
        len(adaptor.get_atoms(s, msonable=False)) for s in structure_list
    )
    start_time = time.time()

    init_state = ts.initialize_state(
        [adaptor.get_atoms(structure, msonable=False) for structure in structure_list],
        device=device,
        dtype=dtype,
    )
    if apply_fix_symmetry:
        constraints = ts_FixSymmetry.from_state(init_state, symprec=1e-5, refine_symmetry_state=False)
        init_state.constraints = [constraints]

    max_memory_scaler = get_max_memory_scaler(calc, init_state, max_memory_padding)

    state, max_memory_scaler = optimize_with_oom_fallback(
        calc,
        device,
        max_memory_scaler,
        system=init_state,
        optimizer=ts.Optimizer.fire,
        convergence_fn=force_convergence_fn,
        max_steps=short_steps_per_atom * num_atoms,
    )
    update_cached_raw_scaler(calc, max_memory_scaler, max_memory_padding)
    short_stage_elapsed = time.time() - start_time

    state_atoms = state.to_atoms()
    state_energies = state.energy
    pop_indices = []

    for idx, atoms in enumerate(state_atoms):
        if not check_distance_tolerance(atoms, tol_matrix):
            result = _build_relax_result_for_batch(
                atoms,
                state_energies[idx],
                adaptor,
                start_time,
                t_gen_list[idx],
                status='distance_violation_initial',
                elapsed_time=short_stage_elapsed,
            )
            relax_results.append(result)
            pop_indices.append(idx)
        else:
            relax_results.append(None)

    state.pop(pop_indices)

    # Release cached allocations between the short and final relaxation stages.
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if len(state.energy.detach().cpu().numpy()) > 0:
        final_state, max_memory_scaler = optimize_with_oom_fallback(
            calc,
            device,
            max_memory_scaler,
            system=state,
            optimizer=ts.Optimizer.fire,
            convergence_fn=force_convergence_fn,
            max_steps=steps_per_atom * num_atoms,
        )
        update_cached_raw_scaler(calc, max_memory_scaler, max_memory_padding)
        final_stage_elapsed = time.time() - start_time

        final_state_atoms = final_state.to_atoms()
        final_state_energies = final_state.energy

        none_indices = [i for i, res in enumerate(relax_results) if res is None]
        assert len(none_indices) == len(final_state_atoms), (
            "Mismatch between remaining states and placeholder results"
        )

        final_converged = force_convergence_fn(final_state).detach().cpu().tolist()

        for idx, atoms in enumerate(final_state_atoms):
            status = _determine_relaxation_status(
                atoms,
                tol_matrix,
                bool(final_converged[idx]),
            )
            result = _build_relax_result_for_batch(
                atoms,
                final_state_energies[idx],
                adaptor,
                start_time,
                t_gen_list[none_indices[idx]],
                status=status,
                elapsed_time=final_stage_elapsed,
            )
            relax_results[none_indices[idx]] = result

    return relax_results


def _batch_relax_with_symmetry_fallback(
    structure_list: List[Structure],
    t_gen_list: List[float],
    calc,
    tol_matrix,
    config: Dict[str, Any],
) -> List[RelaxResult]:
    """Retry batch relaxation by isolating structures that fail FixSymmetry."""
    import torch_sim as ts
    from torch_sim.constraints import FixSymmetry as ts_FixSymmetry

    adaptor = AseAtomsAdaptor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    symmetry_ok: list[int] = []
    symmetry_failed: list[int] = []

    for idx, structure in enumerate(structure_list):
        single_state = ts.initialize_state(
            [adaptor.get_atoms(structure, msonable=False)],
            device=device,
            dtype=dtype,
        )
        try:
            ts_FixSymmetry.from_state(single_state, symprec=1e-5, refine_symmetry_state=False)
            symmetry_ok.append(idx)
        except ValueError as exc:
            logger.warning(
                "FixSymmetry failed for structure index %d; relaxing it without "
                "symmetry constraints. Error: %s",
                idx,
                exc,
            )
            symmetry_failed.append(idx)

    merged_results: List[RelaxResult | None] = [None] * len(structure_list)

    if symmetry_ok:
        ok_results = _run_batch_relaxation(
            [structure_list[idx] for idx in symmetry_ok],
            [t_gen_list[idx] for idx in symmetry_ok],
            calc,
            tol_matrix,
            config,
            apply_fix_symmetry=True,
        )
        for idx, result in zip(symmetry_ok, ok_results):
            merged_results[idx] = result

    if symmetry_failed:
        fallback_results = _run_batch_relaxation(
            [structure_list[idx] for idx in symmetry_failed],
            [t_gen_list[idx] for idx in symmetry_failed],
            calc,
            tol_matrix,
            config,
            apply_fix_symmetry=False,
        )
        for idx, result in zip(symmetry_failed, fallback_results):
            merged_results[idx] = result

    if any(result is None for result in merged_results):
        raise RuntimeError("Missing relaxation result after FixSymmetry fallback")
    return merged_results  # type: ignore[return-value]


# =============================================================================
# Helper functions for building RelaxResult
# =============================================================================
def _build_relax_result_for_serial(
    atoms: Atoms,
    adaptor: AseAtomsAdaptor, 
    start_time: float, 
    t_gen: float,
    status: str
) -> RelaxResult:
    """Build a RelaxResult from relaxed atoms (Sequential relaxation).
    
    Args:
        atoms: Relaxed ASE Atoms object.
        adaptor: AseAtomsAdaptor for structure conversion.
        start_time: Relaxation start time for duration calculation.
        t_gen: Structure generation time in seconds.
        status: Status string describing relaxation outcome.
    
    Returns:
        RelaxResult with structure, energy per atom, status, and timing.
    """
    energy_per_atom = atoms.get_potential_energy() / len(atoms)

    atoms.set_constraint()  # Remove constraints before conversion
    atoms.calc = None  # Clear calculator to free memory

    relaxed_structure = adaptor.get_structure(atoms) # Pymatgen structure, cannot save FixSymmetry constraint

    return _build_relax_result(
        relaxed_structure,
        energy_per_atom,
        start_time,
        t_gen,
        status,
    )


def _build_relax_result_for_batch(
    atoms: Atoms,
    energy: float,
    adaptor: AseAtomsAdaptor, 
    start_time: float, 
    t_gen: float,
    status: str,
    elapsed_time: float | None = None,
) -> RelaxResult:
    """Build a RelaxResult from relaxed atoms (Batch relaxation).
    
    Args:
        atoms: Relaxed ASE Atoms object.
        energy: Total energy of the relaxed structure.
        adaptor: AseAtomsAdaptor for structure conversion.
        start_time: Relaxation start time for duration calculation.
        t_gen: Structure generation time in seconds.
        status: Status string describing relaxation outcome.
        elapsed_time: Optional precomputed elapsed wall time for this result.
    
    Returns:
        RelaxResult with structure, energy per atom, status, and timing.
    """
    energy_per_atom = float(energy) / len(atoms)
    relaxed_structure = adaptor.get_structure(atoms) # Pymatgen structure, cannot save FixSymmetry constraint
    return _build_relax_result(
        relaxed_structure,
        energy_per_atom,
        start_time,
        t_gen,
        status,
        elapsed_time=elapsed_time,
    )
