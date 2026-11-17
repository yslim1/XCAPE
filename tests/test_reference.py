from pathlib import Path

import pytest

from xcape import reference
from xcape.reference import (
    _apply_two_theta_range,
    _expand_two_theta_range,
    _resolve_input_file_path,
    ensure_distance_constraint,
    prepare_reference,
)


def _fake_ref_info(**overrides):
    info = {
        'chemical_formula': 'NaCl',
        'lattice_parameters': [5.6, 5.6, 5.6, 90.0, 90.0, 90.0],
        'spg_symbol': 'Fm-3m',
        'spg_num': 225,
        'z': 4,
        'volume': 175.6,
        'material': {'Na': 4, 'Cl': 4},
        'two_theta': [20.0, 30.0, 45.0],
        'intensity': [10.0, 100.0, 40.0],
    }
    info.update(overrides)
    return info


# =============================================================================
# _resolve_input_file_path
# =============================================================================
def test_resolve_input_file_path_direct_file_returned_as_is(tmp_path):
    file_path = tmp_path / 'ref.xml'
    file_path.touch()
    assert _resolve_input_file_path(str(file_path), '*.xml', 'XML') == str(file_path)


def test_resolve_input_file_path_directory_with_one_match(tmp_path):
    (tmp_path / 'ref.xml').touch()
    resolved = _resolve_input_file_path(str(tmp_path), '*.xml', 'XML')
    assert resolved == str(tmp_path / 'ref.xml')


def test_resolve_input_file_path_directory_with_no_match_raises(tmp_path):
    with pytest.raises(ValueError, match='No XML files'):
        _resolve_input_file_path(str(tmp_path), '*.xml', 'XML')


def test_resolve_input_file_path_directory_with_multiple_matches_raises(tmp_path):
    (tmp_path / 'a.xml').touch()
    (tmp_path / 'b.xml').touch()
    with pytest.raises(ValueError, match='a.xml, b.xml'):
        _resolve_input_file_path(str(tmp_path), '*.xml', 'XML')


def test_resolve_input_file_path_nonexistent_path_raises(tmp_path):
    missing = tmp_path / 'does_not_exist'
    with pytest.raises(ValueError, match='does not exist'):
        _resolve_input_file_path(str(missing), '*.xml', 'XML')


# =============================================================================
# _expand_two_theta_range
# =============================================================================
def test_expand_two_theta_range_none_passthrough():
    assert _expand_two_theta_range(None, 0.3) is None


def test_expand_two_theta_range_subtracts_and_adds_tol():
    assert _expand_two_theta_range([10.0, 80.0], 0.5) == (9.5, 80.5)


# =============================================================================
# _apply_two_theta_range
# =============================================================================
def test_apply_two_theta_range_none_returns_unchanged():
    ref_info = {'two_theta': [10.0, 20.0], 'intensity': [1.0, 2.0]}
    assert _apply_two_theta_range(ref_info, None) is ref_info


def test_apply_two_theta_range_filters_pairs_inclusive():
    ref_info = {'two_theta': [5.0, 10.0, 20.0, 30.0], 'intensity': [1.0, 2.0, 3.0, 4.0]}
    result = _apply_two_theta_range(ref_info, (10.0, 20.0))
    assert result['two_theta'] == [10.0, 20.0]
    assert result['intensity'] == [2.0, 3.0]


def test_apply_two_theta_range_filters_out_all_peaks_raises():
    ref_info = {'two_theta': [5.0, 90.0], 'intensity': [1.0, 2.0]}
    with pytest.raises(ValueError, match='No reference peaks remain'):
        _apply_two_theta_range(ref_info, (10.0, 20.0))


def test_apply_two_theta_range_keeps_d_list_aligned():
    # Regression: the 'd' list (present on the XML path) must be cropped in
    # lockstep with two_theta/intensity so it can't go stale/misaligned.
    ref_info = {
        'two_theta': [5.0, 10.0, 20.0, 30.0],
        'intensity': [1.0, 2.0, 3.0, 4.0],
        'd': [17.7, 8.8, 4.4, 3.0],
    }
    result = _apply_two_theta_range(ref_info, (10.0, 20.0))
    assert result['two_theta'] == [10.0, 20.0]
    assert result['intensity'] == [2.0, 3.0]
    assert result['d'] == [8.8, 4.4]


# =============================================================================
# prepare_reference
# =============================================================================
def test_prepare_reference_returns_ref_info_and_updates_config(config_factory, monkeypatch):
    config = config_factory()
    Path(config['input_setting']['xml_path']).touch()

    fake_info = _fake_ref_info()
    monkeypatch.setattr(reference, 'xml_to_info', lambda path, wavelength: fake_info)

    result = prepare_reference(config)

    assert result is fake_info
    assert config['lattice'] == {
        'a': 5.6, 'b': 5.6, 'c': 5.6, 'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0,
    }
    assert config['spg_num'] == 225
    assert config['spg_symbol'] == 'Fm-3m'
    assert config['volume'] == 175.6
    assert config['material'] == {'Na': 4, 'Cl': 4}


def test_prepare_reference_both_xml_and_cif_path_raises(config_factory):
    config = config_factory()
    config['input_setting']['cif_path'] = str(Path(config['input_setting']['xml_path']).parent / 'ref.cif')
    with pytest.raises(ValueError, match='Do not set both'):
        prepare_reference(config)


def test_prepare_reference_missing_wave_length_raises(config_factory):
    config = config_factory()
    del config['XRD_setting']['wave_length']
    with pytest.raises(ValueError, match='wave_length'):
        prepare_reference(config)


def test_prepare_reference_neither_xml_nor_cif_path_raises(config_factory):
    config = config_factory()
    config['input_setting'] = {}
    with pytest.raises(ValueError, match='Either "xml_path" or "cif_path"'):
        prepare_reference(config)


def test_prepare_reference_cif_path_expands_two_theta_range(config_factory, monkeypatch, tmp_path):
    config = config_factory()
    del config['input_setting']['xml_path']
    cif_path = tmp_path / 'ref.cif'
    cif_path.touch()
    config['input_setting']['cif_path'] = str(cif_path)

    fake_info = _fake_ref_info()
    captured = {}

    def fake_cif_to_info(path, wavelength, two_theta_range):
        captured['path'] = path
        captured['wavelength'] = wavelength
        captured['two_theta_range'] = two_theta_range
        return fake_info

    monkeypatch.setattr(reference, 'cif_to_info', fake_cif_to_info)

    result = prepare_reference(config)

    assert result is fake_info
    tol = config['XRD_setting']['two_theta_tol']
    lo, hi = config['XRD_setting']['two_theta_range']
    assert captured['two_theta_range'] == (lo - tol, hi + tol)


def test_prepare_reference_invalid_ref_info_raises_from_validation(config_factory, monkeypatch):
    config = config_factory()
    Path(config['input_setting']['xml_path']).touch()

    incomplete_info = _fake_ref_info()
    del incomplete_info['spg_num']
    monkeypatch.setattr(reference, 'xml_to_info', lambda path, wavelength: incomplete_info)

    with pytest.raises(ValueError, match='spg_num'):
        prepare_reference(config)


# =============================================================================
# ensure_distance_constraint
# =============================================================================
def test_ensure_distance_constraint_uses_existing_config_value(config_factory, monkeypatch):
    config = config_factory()
    config['distance_constraint'] = {'Na-Cl': 2.5}

    def fail_if_called(*args, **kwargs):
        raise AssertionError('get_distance_constraint should not be called')

    monkeypatch.setattr(reference, 'get_distance_constraint', fail_if_called)

    tol_matrix = ensure_distance_constraint(config, calc=None)

    assert tol_matrix.get_tol('Na', 'Cl') == pytest.approx(2.5)


def test_ensure_distance_constraint_generates_when_absent(config_factory, monkeypatch):
    config = config_factory()
    assert 'distance_constraint' not in config

    monkeypatch.setattr(
        reference, 'get_distance_constraint', lambda config, calc: {'Na-Cl': 3.1}
    )

    tol_matrix = ensure_distance_constraint(config, calc=None)

    assert config['distance_constraint'] == {'Na-Cl': 3.1}
    assert tol_matrix.get_tol('Na', 'Cl') == pytest.approx(3.1)
