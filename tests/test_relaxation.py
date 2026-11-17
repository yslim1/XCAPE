import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pyxtal.tolerance import Tol_matrix

from xcape.modules import relaxation
from xcape.utils.torchsim_utils import DEFAULT_MAX_MEMORY_PADDING


class ZeroForceCalculator(Calculator):
    """ASE calculator with zero forces everywhere.

    FIRE checks convergence *before* taking any step, so with zero forces
    the optimizer reports converged=True on step 0 without moving atoms.
    This lets serial_relax's two-stage optimization run instantly, with no
    real MLIP forward pass.
    """
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {'energy': 0.0, 'forces': np.zeros((len(atoms), 3))}


def _two_atom_nacl(d):
    return Atoms('NaCl', positions=[[0, 0, 0], [d, 0, 0]], cell=[10, 10, 10], pbc=True)


# =============================================================================
# _validate_relaxation_inputs
# =============================================================================
def test_validate_relaxation_inputs_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        relaxation._validate_relaxation_inputs([1, 2], [0.0])


def test_validate_relaxation_inputs_passes_on_matching_lengths():
    relaxation._validate_relaxation_inputs([1, 2], [0.0, 0.0])


def test_validate_relaxation_inputs_passes_on_both_empty():
    relaxation._validate_relaxation_inputs([], [])


# =============================================================================
# _get_optimizer_settings
# =============================================================================
def test_get_optimizer_settings_extracts_configured_values(config_factory):
    config = config_factory()
    settings = relaxation._get_optimizer_settings(config)
    assert settings == {
        'fmax': 0.02,
        'steps_per_atom': 20,
        'short_steps_per_atom': 5,
        'fix_symmetry': True,
        'max_memory_padding': 0.8,
    }


def test_get_optimizer_settings_defaults_for_empty_config():
    settings = relaxation._get_optimizer_settings({})
    assert settings == {
        'fmax': relaxation.DEFAULT_FMAX,
        'steps_per_atom': relaxation.DEFAULT_STEPS_PER_ATOM,
        'short_steps_per_atom': relaxation.DEFAULT_SHORT_STEPS_PER_ATOM,
        'fix_symmetry': False,
        'max_memory_padding': DEFAULT_MAX_MEMORY_PADDING,
    }


# =============================================================================
# check_distance_tolerance
# =============================================================================
def test_check_distance_tolerance_true_when_distance_above_tolerance(default_tol_matrix):
    tol = default_tol_matrix.get_tol('Na', 'Cl')
    atoms = _two_atom_nacl(d=tol * 1.5)
    assert relaxation.check_distance_tolerance(atoms, default_tol_matrix) is True


def test_check_distance_tolerance_false_when_distance_below_tolerance(default_tol_matrix):
    tol = default_tol_matrix.get_tol('Na', 'Cl')
    atoms = _two_atom_nacl(d=tol * 0.5)
    assert relaxation.check_distance_tolerance(atoms, default_tol_matrix) is False


def test_check_distance_tolerance_on_real_nacl_structure(nacl_structure, default_tol_matrix):
    atoms = AseAtomsAdaptor().get_atoms(nacl_structure, msonable=False)
    assert relaxation.check_distance_tolerance(atoms, default_tol_matrix) is True

    # A large (but not extreme) tolerance factor already exceeds real Na-Cl
    # distances; a very large factor (e.g. 50x) also works but blows up the
    # neighbor-list cutoff radius and is much slower for no added coverage.
    strict_tol_matrix = Tol_matrix(prototype='atomic', factor=5.0)
    assert relaxation.check_distance_tolerance(atoms, strict_tol_matrix) is False


# =============================================================================
# _determine_relaxation_status
# =============================================================================
def test_determine_relaxation_status_distance_violation_wins_over_converged(default_tol_matrix):
    tol = default_tol_matrix.get_tol('Na', 'Cl')
    atoms = _two_atom_nacl(d=tol * 0.5)
    assert relaxation._determine_relaxation_status(atoms, default_tol_matrix, converged=True) \
        == 'distance_violation_final'


def test_determine_relaxation_status_distance_violation_when_not_converged(default_tol_matrix):
    tol = default_tol_matrix.get_tol('Na', 'Cl')
    atoms = _two_atom_nacl(d=tol * 0.5)
    assert relaxation._determine_relaxation_status(atoms, default_tol_matrix, converged=False) \
        == 'distance_violation_final'


def test_determine_relaxation_status_converged(default_tol_matrix):
    tol = default_tol_matrix.get_tol('Na', 'Cl')
    atoms = _two_atom_nacl(d=tol * 1.5)
    assert relaxation._determine_relaxation_status(atoms, default_tol_matrix, converged=True) \
        == 'converged'


def test_determine_relaxation_status_not_converged(default_tol_matrix):
    tol = default_tol_matrix.get_tol('Na', 'Cl')
    atoms = _two_atom_nacl(d=tol * 1.5)
    assert relaxation._determine_relaxation_status(atoms, default_tol_matrix, converged=False) \
        == 'not_converged'


# =============================================================================
# _build_relax_result
# =============================================================================
def test_build_relax_result_fields_land_correctly(nacl_structure):
    result = relaxation._build_relax_result(
        nacl_structure, -3.5, start_time=time.time(), t_gen=1.25,
        status='converged', elapsed_time=2.0,
    )
    assert result.structure is nacl_structure
    assert result.energy == pytest.approx(-3.5)
    assert result.status == 'converged'
    assert result.t_gen == pytest.approx(1.25)
    assert result.t_rel == pytest.approx(2.0)


def test_build_relax_result_computes_elapsed_time_when_none(nacl_structure):
    before = time.time()
    result = relaxation._build_relax_result(
        nacl_structure, -1.0, start_time=before, t_gen=0.0, status='converged',
    )
    after = time.time()
    assert 0 <= result.t_rel <= (after - before) + 1e-3


# =============================================================================
# serial_relax
# =============================================================================
def test_serial_relax_empty_list_returns_empty():
    assert relaxation.serial_relax([], [], MagicMock(), MagicMock(), {}) == []


def test_serial_relax_raises_on_length_mismatch(nacl_structure, default_tol_matrix, config_factory):
    config = config_factory()
    with pytest.raises(ValueError):
        relaxation.serial_relax(
            [nacl_structure], [0.0, 1.0], ZeroForceCalculator(), default_tol_matrix, config,
        )


def test_serial_relax_converges_with_zero_force_calculator(nacl_structure, default_tol_matrix, config_factory):
    config = config_factory()
    config['optimizer']['fix_symmetry'] = False
    config['optimizer']['max_short_step_per_atom'] = 1
    config['optimizer']['max_step_per_atom'] = 1

    results = relaxation.serial_relax(
        [nacl_structure], [0.5], ZeroForceCalculator(), default_tol_matrix, config,
    )

    assert len(results) == 1
    result = results[0]
    assert result.status == 'converged'
    assert isinstance(result.structure, Structure)
    assert np.isfinite(result.energy)
    assert result.t_gen == pytest.approx(0.5)


def test_serial_relax_flags_distance_violation_initial(nacl_structure, config_factory):
    config = config_factory()
    config['optimizer']['fix_symmetry'] = False
    config['optimizer']['max_short_step_per_atom'] = 1
    config['optimizer']['max_step_per_atom'] = 1

    strict_tol_matrix = Tol_matrix(prototype='atomic', factor=5.0)
    results = relaxation.serial_relax(
        [nacl_structure], [0.0], ZeroForceCalculator(), strict_tol_matrix, config,
    )

    assert len(results) == 1
    assert results[0].status == 'distance_violation_initial'


# =============================================================================
# _build_relax_result_for_batch
# =============================================================================
def test_build_relax_result_for_batch_divides_energy_by_atom_count(nacl_structure):
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(nacl_structure, msonable=False)
    result = relaxation._build_relax_result_for_batch(
        atoms, energy=-16.0, adaptor=adaptor, start_time=time.time(),
        t_gen=0.0, status='converged', elapsed_time=1.0,
    )
    assert result.energy == pytest.approx(-16.0 / len(atoms))
    assert isinstance(result.structure, Structure)


# =============================================================================
# batch_relax: empty input / validation, without touching torch_sim
# =============================================================================
def test_batch_relax_empty_list_returns_empty():
    assert relaxation.batch_relax([], [], MagicMock(), MagicMock(), {}) == []


def test_batch_relax_raises_on_length_mismatch(nacl_structure, default_tol_matrix, config_factory):
    config = config_factory()
    with pytest.raises(ValueError):
        relaxation.batch_relax(
            [nacl_structure], [0.0, 1.0], MagicMock(), default_tol_matrix, config,
        )


# =============================================================================
# batch_relax: dispatch logic, with _run_batch_relaxation/_batch_relax_with_
# symmetry_fallback mocked out so torch_sim is never actually invoked.
# =============================================================================
def test_batch_relax_dispatches_without_fix_symmetry(monkeypatch, nacl_structure, default_tol_matrix, config_factory):
    config = config_factory()
    config['optimizer']['fix_symmetry'] = False
    run_batch = MagicMock(return_value='no-symmetry-result')
    fallback = MagicMock()
    monkeypatch.setattr(relaxation, '_run_batch_relaxation', run_batch)
    monkeypatch.setattr(relaxation, '_batch_relax_with_symmetry_fallback', fallback)

    result = relaxation.batch_relax([nacl_structure], [0.0], MagicMock(), default_tol_matrix, config)

    assert result == 'no-symmetry-result'
    run_batch.assert_called_once()
    assert run_batch.call_args.kwargs['apply_fix_symmetry'] is False
    fallback.assert_not_called()


def test_batch_relax_dispatches_with_fix_symmetry_when_it_succeeds(monkeypatch, nacl_structure, default_tol_matrix, config_factory):
    config = config_factory()
    config['optimizer']['fix_symmetry'] = True
    run_batch = MagicMock(return_value='symmetry-result')
    fallback = MagicMock()
    monkeypatch.setattr(relaxation, '_run_batch_relaxation', run_batch)
    monkeypatch.setattr(relaxation, '_batch_relax_with_symmetry_fallback', fallback)

    result = relaxation.batch_relax([nacl_structure], [0.0], MagicMock(), default_tol_matrix, config)

    assert result == 'symmetry-result'
    assert run_batch.call_args.kwargs['apply_fix_symmetry'] is True
    fallback.assert_not_called()


def test_batch_relax_falls_back_when_fix_symmetry_raises_value_error(monkeypatch, nacl_structure, default_tol_matrix, config_factory):
    config = config_factory()
    config['optimizer']['fix_symmetry'] = True
    run_batch = MagicMock(side_effect=ValueError("incompatible symmetry"))
    fallback = MagicMock(return_value='fallback-result')
    monkeypatch.setattr(relaxation, '_run_batch_relaxation', run_batch)
    monkeypatch.setattr(relaxation, '_batch_relax_with_symmetry_fallback', fallback)

    result = relaxation.batch_relax([nacl_structure], [0.0], MagicMock(), default_tol_matrix, config)

    assert result == 'fallback-result'
    run_batch.assert_called_once()
    assert run_batch.call_args.kwargs['apply_fix_symmetry'] is True
    fallback.assert_called_once()
