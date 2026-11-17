"""
Data types used in the XCAPE package.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Callable, Literal, TypedDict

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.structure_matcher import StructureMatcher


class Environment(TypedDict):
    output_path: Path
    generator: Callable[..., Any]
    mlip: Any
    xrd_calc: XRDCalculator
    structure_matcher: StructureMatcher


class ReferenceInfo(TypedDict):
    two_theta: list[float]
    intensity: list[float]
    chemical_formula: str
    lattice_parameters: list[float]
    spg_symbol: str
    spg_num: int
    z: int | float
    volume: float
    material: dict[str, Any]
    wavelength: Any


OriginType = Literal[
    'random',
    'kept_best',
    'swap_same_mult',
    'split_site',
    'merge_site',
    'swap_diff_elem',
    'site_reconfig',
    'resample_pos',
    'multi_resample_pos',
    'spg_mismatch',
    'max_attempts_reached',
    'non-sym',
]


@dataclass
class RelaxResult:
    structure: Any # Structure object (e.g., pymatgen Structure)
    energy: float # Final energy after relaxation
    status: str # Status of the relaxation
    t_rel: float # Relaxation time in seconds
    t_gen: Optional[float] = None # Generation time in seconds
    r_factor: Optional[float] = None # R-factor for the structure
    wasserstein: Optional[float] = None # Wasserstein distance for the structure
    gen: Optional[int] = None
    pop: Optional[int] = None
    origin_type: OriginType = 'random' # How this candidate was created
    parent: str = 'none' # Parent candidate identifier (e.g., "3_12")


def copy_relax_result(relax_result: RelaxResult) -> RelaxResult:
    # Shallow copy: the structure is copied, all other (immutable) fields are reused.
    return RelaxResult(
        structure=relax_result.structure.copy(),
        energy=relax_result.energy,
        status=relax_result.status,
        t_rel=relax_result.t_rel,
        t_gen=relax_result.t_gen,
        r_factor=relax_result.r_factor,
        wasserstein=relax_result.wasserstein,
        gen=relax_result.gen,
        pop=relax_result.pop,
        origin_type=relax_result.origin_type,
        parent=relax_result.parent
    )
