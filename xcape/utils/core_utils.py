"""Utility functions for XCAPE.

This module provides helper functions for lattice parameter handling,
structure manipulation, configuration management, and file I/O.
"""
import math
import csv
import logging
import random
from importlib.resources import files
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import yaml
from pymatgen.core import Composition, Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Poscar
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pyxtal.symmetry import Group
from pyxtal.tolerance import Tol_matrix

# =============================================================================
# Constants
# =============================================================================
AVOGADRO = 6.022e23
ANGSTROM3_TO_CM3 = 1e-24
DEFAULT_NEIGHBOR_CUTOFF = 5.0  # Å, for distance validity check
POSCAR_HEADER_LINES = 8


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        path: Path to the YAML configuration file.
    
    Returns:
        Configuration dictionary.
    
    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def dump_yaml(save_path: str, config: Dict[str, Any]) -> None:
    """Dump a dictionary to a YAML file.
    
    Args:
        save_path: Path to save the YAML file.
        config: Dictionary to serialize.
    """
    with open(save_path, 'w') as file:
        yaml.dump(config, file, sort_keys=False)


def set_global_seeds(seed: Optional[int]) -> None:
    """Seed the global RNGs used across the XCAPE workflow.

    Seeds Python's ``random``, NumPy's global RNG, and (if importable) Torch,
    which together cover PyXtal structure generation, mutation, Boltzmann
    parent sampling, and the melt-quench MD used for distance constraints.

    Note: exact run-to-run reproducibility is only achievable for the
    CPU/structure-generation portions of the pipeline. MLIP relaxation on GPU
    (SevenNet / torch-sim) may still be non-deterministic because of
    non-deterministic CUDA kernels; seeding makes the search far more
    repeatable but is not a hard determinism guarantee on GPU.

    Args:
        seed: Integer seed. If ``None``, no seeding is performed and the run
            uses fresh entropy (non-reproducible).
    """
    if seed is None:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        logging.getLogger(__name__).debug(
            "torch not importable; skipping torch seeding."
        )
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_ltype(lattice: List[float]) -> str:
    """Determine lattice type from lattice parameters.
    
    Args:
        lattice: List of 6 lattice parameters [a, b, c, alpha, beta, gamma].
    
    Returns:
        Lattice type string: 'cubic', 'trigonal', 'tetragonal', 'hexagonal',
        'orthorhombic', 'monoclinic', or 'triclinic'.
    
    Raises:
        ValueError: If lattice parameters don't match any known type.
    """
    a, b, c, alpha, beta, gamma = lattice

    def _eq(x: float, y: float) -> bool:
        return math.isclose(x, y)

    ab = _eq(a, b)
    bc = _eq(b, c)

    if ab and bc:  # a == b == c
        if _eq(alpha, 90) and _eq(beta, 90) and _eq(gamma, 90):
            return 'cubic'
        if _eq(alpha, beta) and _eq(beta, gamma) and not _eq(gamma, 90):
            return 'trigonal'

    elif ab and not bc:  # a == b != c
        if _eq(alpha, 90) and _eq(beta, 90) and _eq(gamma, 90):
            return 'tetragonal'
        if _eq(alpha, 90) and _eq(beta, 90) and _eq(gamma, 120):
            return 'hexagonal'

    else:  # a != b (covers a!=b==c and a!=b!=c)
        if _eq(alpha, 90) and _eq(beta, 90) and _eq(gamma, 90):
            return 'orthorhombic'
        if _eq(alpha, 90) and _eq(gamma, 90) and not _eq(beta, 90):
            return 'monoclinic'
        if not _eq(alpha, 90) or not _eq(beta, 90) or not _eq(gamma, 90):
            return 'triclinic'

    raise ValueError(
        f"Cannot determine lattice type for parameters: "
        f"a={a}, b={b}, c={c}, alpha={alpha}, beta={beta}, gamma={gamma}"
    )


def get_volume_from_lattice(lattice: List[float]) -> float:
    """Calculate unit cell volume from lattice parameters.
    
    Args:
        lattice: List of 6 parameters [a, b, c, alpha, beta, gamma].
            Lengths in Å, angles in degrees.
    
    Returns:
        Unit cell volume in Å³.
    """
    a, b, c, alpha, beta, gamma = lattice
    
    # Convert angles to radians
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)
    gamma_rad = math.radians(gamma)

    cos_alpha = math.cos(alpha_rad)
    cos_beta = math.cos(beta_rad)
    cos_gamma = math.cos(gamma_rad)

    volume = a * b * c * math.sqrt(
        1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 +
        2 * cos_alpha * cos_beta * cos_gamma
    )
    return volume


def get_Z_from_density(comp: str, density: float, volume: float) -> int:
    """Calculate formula units per unit cell (Z) from density.
    
    Args:
        comp: Chemical formula string (e.g., 'Fe2O3').
        density: Density in g/cm³.
        volume: Unit cell volume in Å³.
    
    Returns:
        Number of formula units per unit cell (rounded to integer).
    """
    comp_obj = Composition(comp)
    formula_weight = comp_obj.weight

    volume_cm3 = volume * ANGSTROM3_TO_CM3
    z_value = (density * volume_cm3) / (formula_weight / AVOGADRO)

    return round(z_value)


def copy_structure(structure: pyxtal) -> pyxtal:
    """Create a deep copy of a PyXtal structure.
    
    Uses pymatgen conversion as a workaround for PyXtal's buggy copy method.
    
    Args:
        structure: PyXtal structure object to copy.
    
    Returns:
        New PyXtal structure object with identical content.
    """
    pmg_structure = structure.to_pymatgen()
    new_structure = pyxtal()
    new_structure.from_seed(pmg_structure)
    return new_structure


def check_distance_validity(
    structure: pyxtal, 
    tol_matrix: Tol_matrix
) -> List[Tuple[Element, Element, float, float]]:
    """Find atom pairs with unphysically short distances.
    
    Args:
        structure: PyXtal structure object to check.
        tol_matrix: Tolerance matrix for minimum allowed distances.
    
    Returns:
        List of tuples (element_i, element_j, distance, cutoff) for each
        pair that violates the distance constraint. Empty list if valid.
    """
    pmg_structure = structure.to_pymatgen()
    overlaps: List[Tuple[Element, Element, float, float]] = []

    for site_i in pmg_structure:
        atomic_num_i = Element(site_i.specie.symbol).number
        neighbors = pmg_structure.get_neighbors(site_i, r=DEFAULT_NEIGHBOR_CUTOFF)

        for neighbor in neighbors:
            site_j = pmg_structure[neighbor.index]
            atomic_num_j = Element(site_j.specie.symbol).number
            cutoff = tol_matrix.get_tol(atomic_num_i, atomic_num_j)

            if neighbor.nn_distance < cutoff:
                overlaps.append((
                    site_i.specie, 
                    site_j.specie, 
                    neighbor.nn_distance, 
                    cutoff
                ))

    return overlaps


def generate_random_crystal(
    spg_number: int, 
    lattice_params: List[float], 
    comp_dict: Dict[str, int], 
    tol_matrix: Tol_matrix
) -> pyxtal:
    """Generate a random crystal structure using PyXtal.
    
    Args:
        spg_number: Space group number (1-230).
        lattice_params: List of 6 parameters [a, b, c, alpha, beta, gamma].
        comp_dict: Dictionary mapping element symbols to atom counts.
        tol_matrix: Tolerance matrix for distance constraints.
    
    Returns:
        Generated PyXtal structure object.
    """
    ltype = get_ltype(lattice_params)
    lattice = Lattice.from_para(*lattice_params, ltype=ltype)
    spg = Group(spg_number)

    crystal = pyxtal()
    crystal.from_random(
        dim=3,
        group=spg,
        lattice=lattice,
        species=list(comp_dict.keys()),
        numIons=list(comp_dict.values()),
        tm=tol_matrix
    )

    return crystal


def poscar_text_from_pymatgen(structure: Structure, gen: int, pop: int) -> str:
    """Convert pymatgen Structure to POSCAR format string.
    
    Args:
        structure: Pymatgen Structure object.
        gen: Generation index (for comment line).
        pop: Population index (for comment line).
    
    Returns:
        POSCAR format string with formatted coordinates.
    """
    poscar = Poscar(structure, comment=f'Gen: {gen} / Pop: {pop}')
    lines = poscar.get_str().splitlines()
    
    formatted_lines = []
    for idx, line in enumerate(lines):
        if idx < POSCAR_HEADER_LINES:
            formatted_lines.append(line)
        else:
            parts = line.split()
            if len(parts) >= 3:
                formatted_lines.append(
                    f"{float(parts[0]):19.15f}"
                    f"{float(parts[1]):19.15f}"
                    f"{float(parts[2]):19.15f}"
                )
    
    return "\n".join(formatted_lines) + '\n'


def update_general_info(config: Dict[str, Any], general_info: Dict[str, Any]) -> None:
    """Update config dictionary with crystallographic reference data.
    
    Args:
        config: Configuration dictionary to update (modified in-place).
        general_info: Reference data containing lattice_parameters, spg_num,
            volume, and material composition.
    """
    lattice = general_info['lattice_parameters']
    config['lattice'] = {
        'a': lattice[0],
        'b': lattice[1],
        'c': lattice[2],
        'alpha': lattice[3],
        'beta': lattice[4],
        'gamma': lattice[5],
    }
    for key in ['spg_num', 'spg_symbol', 'volume', 'material']:
        config[key] = general_info[key]


def normalize_hm_symbol(symbol: str) -> str:
    return ''.join(symbol.split()).replace('_', '').lower()


def hm_symbol_to_hall_number(symbol: str, number: int) -> int:
    """Return the Hall number for a Hermann-Mauguin space group symbol."""
    normalized_symbol = normalize_hm_symbol(symbol)
    HM_FULL_CSV = files('pyxtal').joinpath('database/HM_Full.csv')
    with open(HM_FULL_CSV, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if normalize_hm_symbol(row['Symbol']) == normalized_symbol and int(row['Spg_num']) == number:
                return int(row['Hall'])

           # Handle cases where the symbol may have a '1' suffix (e.g., for spacegroup 15, A12/a1 can be written as A2/a)
            if normalize_hm_symbol(row['Symbol']).replace('1', '') == normalized_symbol and int(row['Spg_num']) == number:
                return int(row['Hall'])
            
    raise ValueError(f"Space group symbol '{symbol}' not found in HM_Full.csv")
