"""Shared fixtures for the XCAPE test suite."""
import copy

import pytest
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pyxtal.symmetry import Group
from pyxtal.tolerance import Tol_matrix

from xcape.utils.types import RelaxResult

CUBIC_LATTICE_PARAMS = [5.6, 5.6, 5.6, 90.0, 90.0, 90.0]


@pytest.fixture
def nacl_pyxtal():
    """Deterministic rock-salt NaCl structure (Fm-3m, all sites dof=0)."""
    lattice = Lattice.from_para(*CUBIC_LATTICE_PARAMS, ltype='cubic')
    spg = Group(225)
    tm = Tol_matrix(prototype='atomic', factor=0.5)
    crystal = pyxtal()
    crystal.from_random(
        dim=3, group=spg, lattice=lattice,
        species=['Na', 'Cl'], numIons=[4, 4], tm=tm,
    )
    return crystal


@pytest.fixture
def nacl_structure(nacl_pyxtal):
    """Pymatgen Structure for the NaCl fixture above."""
    return nacl_pyxtal.to_pymatgen()


@pytest.fixture
def default_tol_matrix():
    return Tol_matrix(prototype='atomic', factor=0.5)


@pytest.fixture
def movable_pyxtal():
    """P1 Na2Cl2 structure; all sites are Wyckoff 1a with dof=3 (fully free)."""
    lattice = Lattice.from_para(8.0, 8.0, 8.0, 90.0, 90.0, 90.0, ltype='cubic')
    spg = Group(1)
    tm = Tol_matrix(prototype='atomic', factor=0.5)
    crystal = pyxtal()
    crystal.from_random(
        dim=3, group=spg, lattice=lattice,
        species=['Na', 'Cl'], numIons=[2, 2], tm=tm,
    )
    return crystal


@pytest.fixture
def movable_structure(movable_pyxtal):
    """Pymatgen Structure for the P1 fixture above."""
    return movable_pyxtal.to_pymatgen()


@pytest.fixture
def make_relax_result():
    """Factory for RelaxResult objects with sensible defaults."""
    def _make(structure=None, energy=-1.0, status='converged', **overrides):
        kwargs = dict(
            structure=structure,
            energy=energy,
            status=status,
            t_rel=1.0,
        )
        kwargs.update(overrides)
        return RelaxResult(**kwargs)
    return _make


@pytest.fixture
def sample_ref_info():
    """Minimal ReferenceInfo-like dict for XRD/reference-related tests."""
    return {
        'chemical_formula': 'NaCl',
        'lattice_parameters': CUBIC_LATTICE_PARAMS,
        'spg_symbol': 'Fm-3m',
        'spg_num': 225,
        'z': 4,
        'volume': 175.6,
        'material': {'Na': 4, 'Cl': 4},
        'two_theta': [20.0, 30.0, 45.0],
        'intensity': [10.0, 100.0, 40.0],
        'wavelength': 'CuKa',
    }


@pytest.fixture
def sample_config(tmp_path):
    """Full config dict satisfying validate_config's requirements."""
    return {
        'output_path': str(tmp_path / 'run'),
        'input_setting': {
            'xml_path': str(tmp_path / 'ref.xml'),
        },
        'calc_dist_tol': True,
        'dist_tol_scale': 0.75,
        'dist_tol_limit': 1.2,
        'MLIP_setting': {
            'model': '7net-omni',
            'modal': 'mpa',
            'backend': 'flash',
        },
        'XRD_setting': {
            'wave_length': 'CuKa',
            'two_theta_tol': 0.3,
            'two_theta_range': [1.0, 90.0],
            'normalize': 'sum',
        },
        'optimizer': {
            'type': 'batch',
            'ftol': 0.02,
            'max_short_step_per_atom': 5,
            'max_step_per_atom': 20,
            'fix_symmetry': True,
            'max_memory_padding': 0.8,
        },
        'structure': {
            'generation': 5,
            'population': 5,
            'random_ratio': 0.1,
            'early_stopping_threshold': 0.002,
            'mutation': {
                'selection_method': {
                    'parent_metric': 'energy',
                    'weighted_parent_ratio': 0.8,
                    'temperature': 1.0,
                    'temperature_decay': 0.9,
                    'min_temperature': 0.5,
                },
                'max_mutation_attempts': 100,
                'mutation_ratios': {
                    'swap_same_mult_ratio': 0.25,
                    'split_site_ratio': 0.1,
                    'merge_site_ratio': 0.1,
                    'swap_diff_elem_ratio': 0.15,
                    'site_reconfig_ratio': 0.2,
                    'resample_pos_ratio': 0.1,
                    'multi_resample_pos_ratio': 0.1,
                    'max_attempts': 100,
                },
            },
            'best_metric': {
                'energy': {'enabled': True, 'delta': 0.1},
                'r_factor': {'enabled': True, 'delta': 0.2},
                'wasserstein': {'enabled': True, 'delta': 0.5},
            },
            'dedup': {
                'keep_max': 25,
                'energy': 0.005,
                'r_factor': 0.05,
                'wasserstein': 0.5,
                'mix_ratios': {
                    'r_factor': 0.4,
                    'energy': 0.4,
                    'wasserstein': 0.2,
                },
                'structure_matcher': {
                    'ltol': 0.2,
                    'stol': 0.3,
                    'angle_tol': 5,
                    'primitive_cell': False,
                },
            },
        },
    }


@pytest.fixture
def config_factory(sample_config):
    """Return a deep-copied sample_config so tests can mutate it freely."""
    def _make():
        return copy.deepcopy(sample_config)
    return _make
