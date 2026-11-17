import pickle

import pytest
from ase import Atoms
from ase.io.trajectory import Trajectory

from xcape.modules import distance_constraint as dc


# =============================================================================
# _euclidean_distance / _distance_with_pbc
# =============================================================================
def test_euclidean_distance():
    assert dc._euclidean_distance([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)


def test_distance_with_pbc_picks_minimum_image():
    # Points near opposite faces of a 10 A cubic cell are actually close
    # through the periodic image, not far apart as naive distance suggests.
    pos_a = [0.1, 0.1, 0.1]
    pos_b = [9.9, 9.9, 9.9]
    naive = dc._euclidean_distance(pos_a, pos_b)
    pbc = dc._distance_with_pbc(pos_a, pos_b, cell_length=10.0)
    assert pbc < naive
    assert pbc == pytest.approx(0.2 * 3 ** 0.5)


def test_distance_with_pbc_no_wraparound_needed():
    pos_a = [1.0, 1.0, 1.0]
    pos_b = [3.0, 1.0, 1.0]
    assert dc._distance_with_pbc(pos_a, pos_b, cell_length=10.0) == pytest.approx(2.0)


# =============================================================================
# calc_D
# =============================================================================
def _write_trajectory(path, frames):
    traj = Trajectory(str(path), 'w')
    for positions in frames:
        atoms = Atoms('NaCl', positions=positions, cell=[10, 10, 10], pbc=True)
        traj.write(atoms)
    traj.close()


def test_calc_D_returns_finite_nonnegative_float(tmp_path):
    frames = [[[0.1 * i, 0.0, 0.0], [5.0, 5.0, 5.0]] for i in range(10)]
    traj_path = tmp_path / 'md.traj'
    _write_trajectory(traj_path, frames)

    D = dc.calc_D(str(traj_path))
    assert isinstance(D, float)
    assert D >= 0.0
    assert D == D  # not NaN


def test_calc_D_stationary_atoms_gives_zero():
    frames = [[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]] for _ in range(5)]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        traj_path = f'{d}/md.traj'
        _write_trajectory(traj_path, frames)
        D = dc.calc_D(traj_path)
    assert D == pytest.approx(0.0)


# =============================================================================
# check_shortest_distance
# =============================================================================
def test_check_shortest_distance_reports_minimum_per_pair(tmp_path):
    traj_path = tmp_path / 'md.traj'
    frames = [
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ]
    _write_trajectory(traj_path, frames)

    result = dc.check_shortest_distance(str(traj_path), {'dist_tol_limit': 0})
    assert result == {'Cl-Na': pytest.approx(1.0)}


def test_check_shortest_distance_clamps_to_tol_limit(tmp_path):
    traj_path = tmp_path / 'md.traj'
    frames = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    _write_trajectory(traj_path, frames)

    result = dc.check_shortest_distance(str(traj_path), {'dist_tol_limit': 1.5})
    assert result == {'Cl-Na': pytest.approx(1.5)}


# =============================================================================
# check_mqa
# =============================================================================
def test_check_mqa_no_pickle_returns_false_none(tmp_path):
    config = {'output_path': str(tmp_path)}
    assert dc.check_mqa(config) == (False, None)


def test_check_mqa_reads_existing_pickle(tmp_path):
    mqa_dir = tmp_path / 'mlip_mqa'
    mqa_dir.mkdir()
    payload = {'Na-Cl': 2.0}
    with open(mqa_dir / 'distance_constraint.pkl', 'wb') as f:
        pickle.dump(payload, f)

    config = {'output_path': str(tmp_path)}
    done, result = dc.check_mqa(config)
    assert done is True
    assert result == payload


# =============================================================================
# set_default_tolerance
# =============================================================================
def test_set_default_tolerance_two_elements():
    config = {'material': {'Na': 4, 'Cl': 4}}
    result = dc.set_default_tolerance(config)
    assert set(result.keys()) == {'Na-Na', 'Na-Cl', 'Cl-Cl'}
    assert all(v > 0 for v in result.values())


def test_set_default_tolerance_three_elements_all_unordered_pairs():
    config = {'material': {'Na': 4, 'Cl': 4, 'K': 2}}
    result = dc.set_default_tolerance(config)
    assert set(result.keys()) == {'Na-Na', 'Na-Cl', 'Na-K', 'Cl-Cl', 'Cl-K', 'K-K'}
    assert all(v > 0 for v in result.values())


# =============================================================================
# get_distance_constraint
# =============================================================================
def test_get_distance_constraint_default_branch_applies_limit_and_scale(tmp_path):
    config = {
        'output_path': str(tmp_path),
        'material': {'Na': 4, 'Cl': 4},
        'calc_dist_tol': False,
        'dist_tol_limit': 3.0,
        'dist_tol_scale': 0.5,
    }
    result = dc.get_distance_constraint(config, None)
    baseline = dc.set_default_tolerance(config)
    for key, base_val in baseline.items():
        assert result[key] == pytest.approx(max(base_val, 3.0) * 0.5)


def test_get_distance_constraint_uses_cached_mqa_result(tmp_path):
    mqa_dir = tmp_path / 'mlip_mqa'
    mqa_dir.mkdir()
    with open(mqa_dir / 'distance_constraint.pkl', 'wb') as f:
        pickle.dump({'Na-Cl': 2.0}, f)

    config = {
        'output_path': str(tmp_path),
        'calc_dist_tol': True,
        'dist_tol_limit': 1.0,
        'dist_tol_scale': 2.0,
    }
    result = dc.get_distance_constraint(config, None)
    assert result == {'Na-Cl': pytest.approx(4.0)}


# =============================================================================
# create_rand_structure
# =============================================================================
def test_create_rand_structure_writes_valid_poscar(tmp_path):
    output_path = tmp_path / 'run'
    (output_path / 'mlip_mqa').mkdir(parents=True)
    config = {
        'output_path': str(output_path),
        'lattice': {'a': 10.0, 'b': 10.0, 'c': 10.0, 'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0},
    }
    dc.create_rand_structure(config, ['Na', 'Cl'], [4, 4], target_num=8)

    poscar_path = output_path / 'mlip_mqa' / 'POSCAR_rand'
    assert poscar_path.is_file()
    lines = poscar_path.read_text().splitlines()

    assert lines[5].split() == ['Na', 'Cl']
    assert lines[6].split()[:2] == ['4', '4']

    assert lines[8] == 'Direct'
    coord_lines = lines[9:]
    assert len(coord_lines) == 8
