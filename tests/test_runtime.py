import logging

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.structure_matcher import StructureMatcher

from xcape import runtime
from xcape.runtime import (
    DEFAULT_ANGLE_TOL,
    DEFAULT_LTOL,
    DEFAULT_STOL,
    build_structure_matcher,
    initialize_generation_log,
    setup_environment,
    setup_logging,
)
from xcape.modules.generation import random_generation_pyxtal


# =============================================================================
# build_structure_matcher
# =============================================================================
def test_build_structure_matcher_defaults_when_unset(config_factory):
    config = config_factory()
    del config['structure']['dedup']['structure_matcher']

    matcher = build_structure_matcher(config)

    assert isinstance(matcher, StructureMatcher)
    assert matcher.ltol == DEFAULT_LTOL
    assert matcher.stol == DEFAULT_STOL
    assert matcher.angle_tol == DEFAULT_ANGLE_TOL
    assert matcher._primitive_cell is False


def test_build_structure_matcher_uses_custom_values(config_factory):
    config = config_factory()
    config['structure']['dedup']['structure_matcher'] = {
        'ltol': 0.1, 'stol': 0.15, 'angle_tol': 2.0, 'primitive_cell': True,
    }

    matcher = build_structure_matcher(config)

    assert matcher.ltol == 0.1
    assert matcher.stol == 0.15
    assert matcher.angle_tol == 2.0
    assert matcher._primitive_cell is True


# =============================================================================
# setup_logging
# =============================================================================
def test_setup_logging_creates_log_file(tmp_path):
    setup_logging(tmp_path)
    logging.getLogger('xcape.test_runtime').info('hello')

    log_path = tmp_path / 'xcape.log'
    assert log_path.exists()
    assert 'hello' in log_path.read_text()


# =============================================================================
# initialize_generation_log
# =============================================================================
def test_initialize_generation_log_writes_header_for_new_file(tmp_path):
    log_file = tmp_path / 'generation.log'
    initialize_generation_log(log_file)
    assert log_file.read_text() == runtime.LOG_HEADER


def test_initialize_generation_log_does_not_duplicate_existing_content(tmp_path):
    log_file = tmp_path / 'generation.log'
    log_file.write_text(runtime.LOG_HEADER + 'some data line\n')
    initialize_generation_log(log_file)
    assert log_file.read_text() == runtime.LOG_HEADER + 'some data line\n'


def test_initialize_generation_log_writes_header_for_empty_existing_file(tmp_path):
    log_file = tmp_path / 'generation.log'
    log_file.touch()
    assert log_file.stat().st_size == 0
    initialize_generation_log(log_file)
    assert log_file.read_text() == runtime.LOG_HEADER


# =============================================================================
# setup_environment
# =============================================================================
def test_setup_environment_builds_expected_environment(config_factory, monkeypatch, tmp_path):
    config = config_factory()
    output_path = tmp_path / 'run_output'
    config['output_path'] = str(output_path)
    assert not output_path.exists()

    fake_mlip = object()
    monkeypatch.setattr(runtime, 'load_sevennet_calc', lambda config: fake_mlip)

    env = setup_environment(config)

    assert env['output_path'] == output_path
    assert env['output_path'].exists()
    assert env['mlip'] is fake_mlip
    assert env['generator'] is random_generation_pyxtal
    assert isinstance(env['xrd_calc'], XRDCalculator)
    assert env['xrd_calc'].wavelength == XRDCalculator(wavelength='CuKa').wavelength
    assert isinstance(env['structure_matcher'], StructureMatcher)
