import math

import pytest
from pyxtal.tolerance import Tol_matrix

from xcape.utils import core_utils


@pytest.mark.parametrize('lattice,expected', [
    ([4, 4, 4, 90, 90, 90], 'cubic'),
    ([4, 4, 4, 80, 80, 80], 'trigonal'),
    ([4, 4, 6, 90, 90, 90], 'tetragonal'),
    ([4, 4, 6, 90, 90, 120], 'hexagonal'),
    ([4, 5, 6, 90, 90, 90], 'orthorhombic'),
    ([4, 5, 6, 90, 100, 90], 'monoclinic'),
    ([4, 5, 6, 80, 90, 100], 'triclinic'),
])
def test_get_ltype(lattice, expected):
    assert core_utils.get_ltype(lattice) == expected


def test_set_global_seeds_makes_random_streams_reproducible():
    import random

    import numpy as np

    core_utils.set_global_seeds(1234)
    py_first = [random.random() for _ in range(5)]
    np_first = np.random.rand(5)

    core_utils.set_global_seeds(1234)
    py_second = [random.random() for _ in range(5)]
    np_second = np.random.rand(5)

    assert py_first == py_second
    np.testing.assert_array_equal(np_first, np_second)


def test_set_global_seeds_none_is_noop_and_leaves_rng_advancing():
    import random

    # None must not seed; consecutive draws should differ (not reset).
    core_utils.set_global_seeds(None)
    first = random.random()
    core_utils.set_global_seeds(None)
    second = random.random()
    assert first != second


def test_get_ltype_raises_for_inconsistent_parameters():
    # a == b == c with non-matching angles for cubic/trigonal.
    with pytest.raises(ValueError):
        core_utils.get_ltype([4, 4, 4, 90, 90, 80])


def test_get_volume_from_lattice_cubic():
    volume = core_utils.get_volume_from_lattice([2, 2, 2, 90, 90, 90])
    assert volume == pytest.approx(8.0)


def test_get_volume_from_lattice_matches_general_formula():
    a, b, c, alpha, beta, gamma = 5.0, 6.0, 7.0, 80.0, 85.0, 95.0
    volume = core_utils.get_volume_from_lattice([a, b, c, alpha, beta, gamma])

    cos = lambda deg: math.cos(math.radians(deg))
    expected = a * b * c * math.sqrt(
        1 - cos(alpha) ** 2 - cos(beta) ** 2 - cos(gamma) ** 2
        + 2 * cos(alpha) * cos(beta) * cos(gamma)
    )
    assert volume == pytest.approx(expected)


def test_get_Z_from_density_nacl():
    # NaCl: formula weight ~58.44 g/mol, density 2.17 g/cm^3, conventional cell volume ~179.4 A^3 -> Z=4
    z = core_utils.get_Z_from_density('NaCl', density=2.17, volume=179.4)
    assert z == 4


def test_copy_structure_round_trips_content(nacl_pyxtal):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    copied = core_utils.copy_structure(nacl_pyxtal)
    original_structure = nacl_pyxtal.to_pymatgen()
    copied_structure = copied.to_pymatgen()

    assert copied is not nacl_pyxtal
    assert copied_structure.composition == original_structure.composition
    # The pymatgen seed round-trip may reorder sites and introduce tiny
    # floating-point noise, so compare structural equivalence rather than
    # exact site-by-site equality.
    assert StructureMatcher().fit(copied_structure, original_structure)


def test_check_distance_validity_no_overlaps_for_relaxed_structure(nacl_pyxtal):
    tol_matrix = Tol_matrix(prototype='atomic', factor=0.5)
    overlaps = core_utils.check_distance_validity(nacl_pyxtal, tol_matrix)
    assert overlaps == []


def test_check_distance_validity_flags_overlap_with_strict_tolerance(nacl_pyxtal):
    # An unreasonably large tolerance factor should flag every close neighbor.
    strict_tol_matrix = Tol_matrix(prototype='atomic', factor=5.0)
    overlaps = core_utils.check_distance_validity(nacl_pyxtal, strict_tol_matrix)
    assert len(overlaps) > 0


def test_generate_random_crystal_matches_requested_composition():
    tol_matrix = Tol_matrix(prototype='atomic', factor=0.5)
    crystal = core_utils.generate_random_crystal(
        spg_number=225,
        lattice_params=[5.6, 5.6, 5.6, 90.0, 90.0, 90.0],
        comp_dict={'Na': 4, 'Cl': 4},
        tol_matrix=tol_matrix,
    )
    structure = crystal.to_pymatgen()
    assert structure.composition.get_el_amt_dict() == {'Na': 4, 'Cl': 4}


def test_poscar_text_from_pymatgen_includes_header_and_comment(nacl_structure):
    text = core_utils.poscar_text_from_pymatgen(nacl_structure, gen=3, pop=7)
    lines = text.splitlines()
    assert lines[0] == 'Gen: 3 / Pop: 7'
    assert text.endswith('\n')
    # 8 header lines + one coordinate line per site.
    assert len(lines) == core_utils.POSCAR_HEADER_LINES + len(nacl_structure)


def test_update_general_info_populates_lattice_and_scalars():
    config = {}
    general_info = {
        'lattice_parameters': [5.6, 5.6, 5.6, 90.0, 90.0, 90.0],
        'spg_num': 225,
        'spg_symbol': 'Fm-3m',
        'volume': 175.6,
        'material': {'Na': 4, 'Cl': 4},
    }
    core_utils.update_general_info(config, general_info)

    assert config['lattice'] == {
        'a': 5.6, 'b': 5.6, 'c': 5.6, 'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0,
    }
    assert config['spg_num'] == 225
    assert config['spg_symbol'] == 'Fm-3m'
    assert config['volume'] == 175.6
    assert config['material'] == {'Na': 4, 'Cl': 4}


@pytest.mark.parametrize('symbol,expected', [
    ('Fm-3m', 'fm-3m'),
    ('F m -3 m', 'fm-3m'),
    ('F_m_-3_m', 'fm-3m'),
])
def test_normalize_hm_symbol(symbol, expected):
    assert core_utils.normalize_hm_symbol(symbol) == expected


def test_hm_symbol_to_hall_number_finds_known_symbol():
    # Fm-3m (space group 225) has Hall number 523 in pyxtal's HM_Full.csv.
    hall_number = core_utils.hm_symbol_to_hall_number('Fm-3m', 225)
    assert isinstance(hall_number, int)
    assert hall_number > 0


def test_hm_symbol_to_hall_number_raises_for_unknown_symbol():
    with pytest.raises(ValueError, match='not found'):
        core_utils.hm_symbol_to_hall_number('Not-A-Real-Symbol', 225)


def test_load_config_round_trips_with_dump_yaml(tmp_path):
    data = {'a': 1, 'b': {'c': [1, 2, 3]}}
    path = tmp_path / 'config.yaml'
    core_utils.dump_yaml(str(path), data)
    loaded = core_utils.load_config(str(path))
    assert loaded == data


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        core_utils.load_config(str(tmp_path / 'missing.yaml'))
