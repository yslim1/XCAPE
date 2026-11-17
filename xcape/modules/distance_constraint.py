#!/usr/bin/env python3
"""Distance constraint calculation module for XCAPE.

This module provides functions to determine minimum interatomic distance
constraints via melt-quench-anneal (MQA) molecular dynamics simulations.
"""
from pathlib import Path
import logging
import math
import datetime
import random
import numpy as np
from typing import Dict, Any, Union, List
import pickle

from pyxtal.tolerance import Tol_matrix

from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase import units

from ..utils.core_utils import get_volume_from_lattice

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================
DEFAULT_MELTING_TEMP = 4200  # Initial temperature for melting simulation (K)
TEMP_STEP = 800  # Temperature step for melting point search (K)
MIN_TEMP = 1000  # Minimum temperature for melting point search (K)
DIFFUSION_THRESHOLD = 4e-9  # Diffusion coefficient threshold for liquid state
DEFAULT_DIST_TOLERANCE = 1.5  # Default minimum interatomic distance (Å)
DEFAULT_DIST_SCALE_FACTOR = 1.5  # Scale factor for default distance tolerance
DEFAULT_FRICTION = 0.02  # Langevin friction coefficient
TIMESTEP_FS = 2  # MD timestep in femtoseconds
TRAJ_WRITE_INTERVAL = 5  # MD steps between trajectory frames used for calc_D
FS_TO_PS = 1e-3  # Femtoseconds -> picoseconds conversion
TARGET_ATOM_COUNT = 300  # Target number of atoms for random structure
MIN_CELL_LENGTH_PARAM = 4.0  # Minimum cell length parameter (Å)


# =============================================================================
# Utility Functions
# =============================================================================
def _euclidean_distance(pos_a: List[float], pos_b: List[float]) -> float:
    """Calculate Euclidean distance between two 3D positions."""
    return sum((x - y) ** 2.0 for x, y in zip(pos_a, pos_b)) ** 0.5


def _distance_with_pbc(pos_a: List[float], pos_b: List[float], cell_length: float) -> float:
    """Calculate minimum distance considering periodic boundary conditions.
    
    Args:
        pos_a: First position [x, y, z].
        pos_b: Second position [x, y, z].
        cell_length: Cubic cell parameter.
    
    Returns:
        Minimum distance considering all periodic images.
    """
    image_offsets = [-1, 0, 1]
    pbc_images = [[i, j, k] for i in image_offsets for j in image_offsets for k in image_offsets]
    b_pbc = [[pos_b[dim] + cell_length * pbc_images[img][dim] for dim in range(3)] 
             for img in range(len(pbc_images))]
    return min(_euclidean_distance(pos_a, b_img) for b_img in b_pbc)


# =============================================================================
# Structure Generation
# =============================================================================
def create_rand_structure(config: Dict[str, Any], elements: List[str], 
                          number: List[int], target_num: int) -> None:
    """Create a random atomic structure and write as POSCAR file.
    
    Generates a random cubic structure with atoms placed respecting minimum
    distance constraints, scaled to match the target composition.

    Args:
        config: Configuration dict containing lattice parameters and output_path.
        elements: List of element symbols (e.g., ['Li', 'O']).
        number: List of atom counts for each element.
        target_num: Target total number of atoms.
    """
    # 1. Calculate stoichiometry with GCD scaling
    GCD = number[0]
    if len(number) >= 2:
        for i in number[1:]:
            GCD = math.gcd(GCD, i)
    smallest = [i//GCD for i in number]
    multiple = target_num//sum(smallest)
    if multiple*sum(smallest) < target_num:
        multiple += 1
    int_number = [i*multiple for i in smallest]
    logger.info("Generating random MQA structure with atom counts %s", int_number)

    # 2. Calculate initial volume from lattice parameters
    a = config['lattice']['a']
    b = config['lattice']['b']
    c = config['lattice']['c']
    alpha = config['lattice']['alpha']
    beta = config['lattice']['beta']
    gamma = config['lattice']['gamma']
    V = get_volume_from_lattice([a, b, c, alpha, beta, gamma])
    logger.info("Initial volume from lattice parameters: %.4f Å³", V)

    # 3. Scale volume according to number of atoms
    mul = multiple / GCD
    V *= mul
    logger.info("Scaled volume for %d atoms: %.4f Å³", sum(int_number), V)

    atomname = []
    for atomi in elements:
        atomname.append(atomi)
    natom = int_number

    # 4. Create random positions with distance constraints
    num_elements = len(atomname)
    total_atoms = sum(natom)
    cutoff = [[DEFAULT_DIST_TOLERANCE for _ in range(num_elements)] 
              for _ in range(num_elements)]

    atom_positions = [[0.0 for _ in range(3)] for _ in range(total_atoms)]
    cell_param = V ** (1.0 / 3.0)
    max_attempts = 100000
    total_attempts = 0
    
    for new_idx in range(total_atoms):
        new_type = sum(new_idx >= sum(natom[:j+1]) for j in range(num_elements))

        placed = False
        while not placed:
            total_attempts += 1
            if total_attempts > max_attempts:
                raise RuntimeError(f"Failed to place atom {new_idx} after {max_attempts} attempts")

            new_position = [cell_param * random.random() for _ in range(3)]
            ok = True
            for existing_idx in range(new_idx):
                existing_type = sum(existing_idx >= sum(natom[:j+1]) for j in range(num_elements))
                dist = _distance_with_pbc(new_position, atom_positions[existing_idx], cell_param)
                if dist < cutoff[new_type][existing_type]:
                    ok = False
                    break

            if ok:
                atom_positions[new_idx] = new_position
                placed = True

    logger.info(
        "Structure generated after %d attempts, cell parameter: %.4f",
        total_attempts, cell_param,
    )

    # 5. Write POSCAR file (converting to fractional coordinates)
    output_path = config['output_path']
    poscar_path = f'{output_path}/mlip_mqa/POSCAR_rand'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert to fractional coordinates
    frac_positions = [[pos[dim] / cell_param for dim in range(3)] 
                      for pos in atom_positions]
    
    with open(poscar_path, 'w') as f:
        f.write(f"{''.join(atomname)}   volume: {V}   {timestamp}\n")
        f.write("   1.0\n")
        f.write(f"{cell_param:21.13f}{0:19.13f}{0:19.13f}\n")
        f.write(f"{0:21.13f}{cell_param:19.13f}{0:19.13f}\n")
        f.write(f"{0:21.13f}{0:19.13f}{cell_param:19.13f}\n")
        f.write('   ' + '   '.join(elements) + "\n")
        f.write('    ' + '    '.join(str(x) for x in natom) + '  \n')
        f.write('Selective dynamics\n')
        f.write('Direct\n')
        
        for i in range(total_atoms - 1):
            pos = frac_positions[i]
            f.write(f"{pos[0]:19.15f}{pos[1]:19.15f}{pos[2]:19.15f}   T   T   T  \n")
        # Last atom fixed
        pos = frac_positions[total_atoms - 1]
        f.write(f"{pos[0]:19.15f}{pos[1]:19.15f}{pos[2]:19.15f}   F   F   F  \n")


def calc_D(
    traj_dir: Union[str, Path],
    timestep_fs: float = TIMESTEP_FS,
    write_interval: int = TRAJ_WRITE_INTERVAL,
) -> float:
    """Calculate diffusion coefficient D from a trajectory file.

    Args:
        traj_dir: Path to trajectory file (ASE readable).
        timestep_fs: MD integration timestep in fs used to generate the
            trajectory. Must match the timestep the trajectory was run with.
        write_interval: Number of MD steps between saved trajectory frames.
            Must match the ``interval`` the trajectory was written with, so the
            physical time per frame is derived rather than hardcoded.

    Returns:
        Diffusion coefficient D (float) in appropriate units.
    """
    traj_dir = str(traj_dir)
    atoms_list = read(traj_dir, index=':')
    N = len(atoms_list)
    if N == 0:
        return 0.0

    idx = 0  # fix atom index
    initial_pos = atoms_list[0].get_positions()[idx]
    shifts = [atoms.get_positions()[idx] for atoms in atoms_list]
    shifts = np.array(shifts)
    shifts -= initial_pos
    shifts = np.expand_dims(shifts, axis=1)
    shifts = np.tile(shifts, (1, len(atoms_list[0]), 1))

    positions = np.array([atoms.get_positions() for atoms in atoms_list])
    positions -= shifts

    # Physical time between two frames separated by `delt` steps of the saved
    # trajectory: delt frames * write_interval steps/frame * timestep (fs) -> ps.
    time_per_frame_ps = write_interval * timestep_fs * FS_TO_PS

    MSD = []
    for delt in range(1, max(2, int(N * 0.7))):
        diffs = positions[delt:] - positions[:-delt]
        squared_distances = np.sum(diffs**2, axis=-1)
        msd_avg = np.mean(squared_distances)
        MSD.append((delt * time_per_frame_ps, msd_avg))

    if len(MSD) == 0:
        return 0.0

    msd = MSD[-1][1]
    time_val = MSD[-1][0]
    D = msd / (6 * time_val) * 1e-8
    return float(D)


def run_mqa(config: Dict[str, Any], calc) -> Dict[str, float]:
    """Run melt-quench MD to determine distance constraints.

    This function is the MD orchestration: it prepares input structures, runs
    premelt / melting routines, generates trajectories, and finally returns the
    shortest interatomic distances computed by analyzers.

    Args:
        config: Configuration dictionary.
        calc: ASE calculator object to be used for MD simulations.

    Returns:
        A dict mapping pair strings (e.g., 'Li-O') to minimum distances (float).
    """
    output_path = Path(config['output_path'])
    (output_path / 'mlip_mqa').mkdir(parents=True, exist_ok=True)

    # 1) Generate initial random structure
    materials = config['material']
    elements = list(materials.keys())
    natoms = list(map(int, materials.values()))
    create_rand_structure(config, elements, natoms, target_num=TARGET_ATOM_COUNT)

    # 2) Initialize ASE calculator
    atoms = read(str(output_path / 'mlip_mqa' / 'POSCAR_rand'), format='vasp')
    atoms.calc = calc

    # 3) Premelt equilibration with Langevin dynamics
    MaxwellBoltzmannDistribution(atoms, temperature_K=DEFAULT_MELTING_TEMP)
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn = Langevin(atoms, timestep=TIMESTEP_FS*units.fs, 
                   temperature_K=DEFAULT_MELTING_TEMP, friction=DEFAULT_FRICTION)
    traj = Trajectory(str(output_path / 'mlip_mqa' / 'premelt.traj'), 'w', atoms)
    dyn.attach(traj.write, interval=50)
    log = MDLogger(dyn, atoms, str(output_path / 'mlip_mqa' / 'premelt.log'), 
                   header=True, stress=True, peratom=True)
    dyn.attach(log, interval=50)
    dyn.run(1500)
    write(str(output_path / 'mlip_mqa' / 'POSCAR_premelt'), atoms, format='vasp')

    # 4) Find melting point by cooling with NVT
    Tm = DEFAULT_MELTING_TEMP
    for T in range(DEFAULT_MELTING_TEMP, MIN_TEMP - 1, -TEMP_STEP):
        dyn = Langevin(atoms, timestep=TIMESTEP_FS*units.fs, 
                       temperature_K=T, friction=DEFAULT_FRICTION)
        traj = Trajectory(str(output_path / 'mlip_mqa' / f'Tm_{T}.traj'), 'w', atoms)
        dyn.attach(traj.write, interval=TRAJ_WRITE_INTERVAL)
        log = MDLogger(dyn, atoms, str(output_path / 'mlip_mqa' / f'Tm_{T}.log'), 
                       header=True, stress=True, peratom=True)
        dyn.attach(log, interval=50)
        dyn.run(500)
        write(str(output_path / 'mlip_mqa' / f'POSCAR_Tm_{T}'), atoms, format='vasp')

        D = calc_D(str(output_path / 'mlip_mqa' / f'Tm_{T}.traj'))
        if D < DIFFUSION_THRESHOLD:
            break
        Tm = T

    # 5) Production melting run at determined Tm
    dyn = Langevin(atoms, timestep=TIMESTEP_FS*units.fs, 
                   temperature_K=Tm, friction=DEFAULT_FRICTION)
    traj = Trajectory(str(output_path / 'mlip_mqa' / 'melt.traj'), 'w', atoms)
    dyn.attach(traj.write, interval=TRAJ_WRITE_INTERVAL)
    log = MDLogger(dyn, atoms, str(output_path / 'mlip_mqa' / 'melt.log'), 
                   header=True, stress=True, peratom=True)
    dyn.attach(log, interval=50)
    dyn.run(4000)
    write(str(output_path / 'mlip_mqa' / 'POSCAR_melt'), atoms, format='vasp')

    # 6) Calculate distance constraints from melt trajectory 
    traj_file = str(output_path / 'mlip_mqa' / 'melt.traj')
    dist_constraint = check_shortest_distance(traj_file, config)
    with open(output_path / 'mlip_mqa' / 'distance_constraint.pkl', 'wb') as f:
        pickle.dump(dist_constraint, f)

    return dist_constraint


def check_mqa(config: Dict[str, Any]) -> tuple:
    """Check if MQA calculation was previously completed.
    
    Args:
        config: Configuration dictionary with 'output_path'.

    Returns:
        Tuple of (done: bool, dist_constraint: dict or None).
    """
    output_path = Path(config['output_path'])
    if (output_path / 'mlip_mqa' / 'distance_constraint.pkl').is_file():
        with open(output_path / 'mlip_mqa' / 'distance_constraint.pkl', 'rb') as f:
            dist_constraint = pickle.load(f)
        return True, dist_constraint
    else:
        return False, None


def check_shortest_distance(traj: Union[str, Path], config: Dict[str, Any]) -> Dict[str, float]:
    """Analyze a trajectory and return shortest interatomic distances per element pair.

    Args:
        traj: Path to ASE trajectory file or filename accepted by ase.io.read.
        config: Configuration containing optional 'dist_tol_limit'.

    Returns:
        Mapping from 'Elem1-Elem2' to minimum observed distance (float).
    """
    traj = str(traj)
    # Prevent too-short distances due to gas formation during MD
    tol_limit = config.get('dist_tol_limit', 0)

    atoms = read(traj, ':')
    shortest: Dict[str, float] = {}
    for atom in atoms:
        symbols = atom.get_chemical_symbols()
        dists = atom.get_all_distances(mic=True)

        n = len(atom)
        for i in range(n):
            for j in range(i + 1, n):
                elem_i = symbols[i]
                elem_j = symbols[j]
                tmp = tuple(sorted((elem_i, elem_j)))
                pair = tmp[0] + "-" + tmp[1]
                d = float(dists[i, j])

                if pair not in shortest or d < shortest[pair]:
                    shortest[pair] = d

    # Clamp unphysically short distances after collecting the true minimum
    for pair in shortest:
        if shortest[pair] < tol_limit:
            shortest[pair] = tol_limit

    return shortest


def get_distance_constraint(config: Dict[str, Any], calc) -> Dict[str, float]:
    """Get distance constraints either from MQA calculation or default values.

    Args:
        config: Configuration dictionary.
        calc: ASE calculator for MD simulations.

    Returns:
        Dictionary mapping element pairs to minimum distances.

    Notes:
        The two sources produce different baselines by design:
        - MQA path: the shortest interatomic distance actually observed in the
          melt trajectory.
        - Default path (``calc_dist_tol: False``): a covalent-radius heuristic
          already scaled by ``DEFAULT_DIST_SCALE_FACTOR`` (1.5) inside
          ``set_default_tolerance`` -- so the default path carries an extra 1.5
          the MQA path does not.

        ``dist_tol_limit`` and ``dist_tol_scale`` are then applied in this
        order, intentionally: ``dist_tol_limit`` floors the *estimate* (guards
        against an implausibly short baseline), and ``dist_tol_scale`` then
        relaxes that floored estimate into the final rejection tolerance. So
        the final value can fall below ``dist_tol_limit`` when
        ``dist_tol_scale < 1`` -- the limit floors the estimate, not the final
        tolerance.
    """
    tol_limit = config.get('dist_tol_limit', 0)
    scale = config.get('dist_tol_scale')

    if config.get('calc_dist_tol', True):
        done, dist_constraint = check_mqa(config)
        if not done:
            dist_constraint = run_mqa(config, calc)

    else:  # Set in default (1.5 * covalent radius)
        dist_constraint = set_default_tolerance(config)

    dist_constraint = dist_constraint.copy()
    for key, value in dist_constraint.items():
        # Floor the estimate first, then relax it into a tolerance (see Notes).
        dist_constraint[key] = max(value, tol_limit)
        if scale is not None:
            dist_constraint[key] *= scale

    return dist_constraint


def set_default_tolerance(config: Dict[str, Any]) -> Dict[str, float]:
    """Set default distance tolerance for all element pairs.
    
    Args:
        config: Configuration dictionary with 'material' key.
        
    Returns:
        Dictionary mapping element pairs to default distance tolerance.
    """
    dist_constraint = {}
    elements = list(config['material'].keys())

    tm = Tol_matrix(prototype='atomic', factor=DEFAULT_DIST_SCALE_FACTOR) # covalent radii * scale factor
    for i, elem_i in enumerate(elements):
        for elem_j in elements[i:]:
            tol = tm.get_tol(elem_i, elem_j)
            dist_constraint[f'{elem_i}-{elem_j}'] = tol

    return dist_constraint
