import xml.etree.ElementTree as ET

import numpy as np
import pytest
from pymatgen.core import Structure, Lattice

from xcape.modules import extraction


# =============================================================================
# parse_formula_counts
# =============================================================================
@pytest.mark.parametrize('formula,expected', [
    ('Li2O', {'Li': 2.0, 'O': 1.0}),
    ('H2O', {'H': 2.0, 'O': 1.0}),
    ('Ca3(PO4)2', {'Ca': 3.0, 'P': 2.0, 'O': 8.0}),
    ('NaCl,H2O', {'Na': 1.0, 'Cl': 1.0, 'H': 2.0, 'O': 1.0}),
    ('NaCl · H2O', {'Na': 1.0, 'Cl': 1.0, 'H': 2.0, 'O': 1.0}),
    ('NaCl * 2H2O', {'Na': 1.0, 'Cl': 1.0, 'H': 4.0, 'O': 2.0}),
])
def test_parse_formula_counts(formula, expected):
    assert extraction.parse_formula_counts(formula) == expected


@pytest.mark.parametrize('formula', ['Fe.', 'Ca.O', '.', 'Na.Cl'])
def test_parse_formula_counts_tolerates_stray_dot(formula):
    # Regression: a lone '.' where a count is expected must not crash
    # (previously float('.') raised ValueError). The dot is treated as no
    # number (implicit multiplier 1.0) and skipped.
    result = extraction.parse_formula_counts(formula)
    assert isinstance(result, dict)
    assert all(count > 0 for count in result.values())


# =============================================================================
# _parse_intensities
# =============================================================================
def test_parse_intensities_normal_values():
    assert extraction._parse_intensities(['100', '50.5']) == [100.0, 50.5]


def test_parse_intensities_skips_empty_and_newline_entries():
    assert extraction._parse_intensities(['100', '\n', '', '50']) == [100.0, 50.0]


def test_parse_intensities_drops_parenthetical_uncertainty():
    # A trailing esd like "(2)" is dropped, not concatenated: "100(2)" is
    # intensity 100 with an uncertainty of 2, so the value is 100.0 (not 1002.0).
    assert extraction._parse_intensities(['100(2)']) == [100.0]
    assert extraction._parse_intensities(['33.0(5)']) == [33.0]


# =============================================================================
# _normalize_lattice_parameters
# =============================================================================
def test_normalize_lattice_parameters_cubic():
    result = extraction._normalize_lattice_parameters('cubic', [5.0, None, None, None, None, None])
    assert result == [5.0, 5.0, 5.0, 90.0, 90.0, 90.0]


def test_normalize_lattice_parameters_tetragonal():
    result = extraction._normalize_lattice_parameters('tetragonal', [5.0, None, 7.0, None, None, None])
    assert result == [5.0, 5.0, 7.0, 90.0, 90.0, 90.0]


def test_normalize_lattice_parameters_hexagonal():
    result = extraction._normalize_lattice_parameters('hexagonal', [5.0, None, 7.0, None, None, None])
    assert result == [5.0, 5.0, 7.0, 90.0, 90.0, 120.0]


def test_normalize_lattice_parameters_orthorhombic():
    result = extraction._normalize_lattice_parameters('orthorhombic', [5.0, 6.0, 7.0, None, None, None])
    assert result == [5.0, 6.0, 7.0, 90.0, 90.0, 90.0]


def test_normalize_lattice_parameters_monoclinic():
    result = extraction._normalize_lattice_parameters('monoclinic', [5.0, 6.0, 7.0, None, 99.0, None])
    assert result == [5.0, 6.0, 7.0, 90.0, 99.0, 90.0]


def test_normalize_lattice_parameters_trigonal_rhombohedral_setting():
    # alpha != 90 -> rhombohedral setting, a=b=c and alpha=beta=gamma.
    result = extraction._normalize_lattice_parameters('trigonal', [5.0, None, None, 80.0, None, None])
    assert result == [5.0, 5.0, 5.0, 80.0, 80.0, 80.0]


def test_normalize_lattice_parameters_trigonal_hexagonal_setting():
    # alpha == 90 -> hexagonal setting.
    result = extraction._normalize_lattice_parameters('trigonal', [5.0, None, 7.0, 90.0, None, None])
    assert result == [5.0, 5.0, 7.0, 90.0, 90.0, 120.0]


# =============================================================================
# find_text_by_tag / _get_required_text / _get_required_float / _get_optional_float
# =============================================================================
def test_find_text_by_tag_returns_all_matches():
    root = ET.fromstring('<root><da>1.0</da><da>2.0</da></root>')
    assert extraction.find_text_by_tag(root, 'da') == ['1.0', '2.0']


def test_find_text_by_tag_missing_tag_returns_empty_list():
    root = ET.fromstring('<root></root>')
    assert extraction.find_text_by_tag(root, 'missing') == []


def test_get_required_text_returns_first_value():
    root = ET.fromstring('<root><name>NaCl</name></root>')
    assert extraction._get_required_text(root, 'name') == 'NaCl'


def test_get_required_text_missing_raises():
    root = ET.fromstring('<root></root>')
    with pytest.raises(ValueError):
        extraction._get_required_text(root, 'name')


def test_get_required_float_strips_parenthesized_esd():
    root = ET.fromstring('<root><cell_a>10.0(2)</cell_a></root>')
    assert extraction._get_required_float(root, 'cell_a') == pytest.approx(10.0)


def test_get_required_float_plain_value():
    root = ET.fromstring('<root><cell_a>10.5</cell_a></root>')
    assert extraction._get_required_float(root, 'cell_a') == pytest.approx(10.5)


def test_get_optional_float_returns_none_when_missing():
    root = ET.fromstring('<root></root>')
    assert extraction._get_optional_float(root, 'cell_b') is None


def test_get_optional_float_returns_value_when_present():
    root = ET.fromstring('<root><cell_b>6.0</cell_b></root>')
    assert extraction._get_optional_float(root, 'cell_b') == pytest.approx(6.0)


# =============================================================================
# xml_to_info
# =============================================================================
def _write_xml(tmp_path, da_values=('3.99', '2.82'), intensity_values=('100', '50')):
    # Mirror the real ICDD PDF-card layout: each reflection is a container
    # element under <stick_series> that nests its own <da> and inner
    # <intensity> together (the container itself is also tagged <intensity>
    # with blank text, exactly as the cards do). da_values / intensity_values
    # are zipped positionally; an index present in only one list yields a
    # reflection missing that value.
    n = max(len(da_values), len(intensity_values))
    reflections = []
    for i in range(n):
        parts = []
        if i < len(da_values):
            parts.append(f'<da>{da_values[i]}</da>')
        if i < len(intensity_values):
            parts.append(f'<intensity>{intensity_values[i]}</intensity>')
        reflections.append(f'<intensity>\n{"".join(parts)}</intensity>')
    stick_series = f'<stick_series>{"".join(reflections)}</stick_series>'
    xml_text = f"""<root>
        <empirical_formula>NaCl</empirical_formula>
        <z>4</z>
        <spgr>F m -3 m (225)</spgr>
        <pdf_number>00-001-2345</pdf_number>
        <cell_vol>175.616</cell_vol>
        <xstal_system>cubic</xstal_system>
        <cell_a>5.6</cell_a>
        <graphs>{stick_series}</graphs>
    </root>"""
    xml_path = tmp_path / 'ref.xml'
    xml_path.write_text(xml_text)
    return str(xml_path)


def test_xml_to_info_parses_all_fields(tmp_path):
    xml_path = _write_xml(tmp_path)
    info = extraction.xml_to_info(xml_path)

    assert info['chemical_formula'] == 'NaCl'
    assert info['z'] == pytest.approx(4.0)
    assert info['material'] == {'Na': 4, 'Cl': 4}
    assert info['spg_symbol'] == 'F m -3 m'
    assert info['spg_num'] == 225
    assert info['pdf_number'] == '00-001-2345'
    assert info['volume'] == pytest.approx(175.616)
    assert info['lattice_parameters'] == [5.6, 5.6, 5.6, 90.0, 90.0, 90.0]
    assert info['d'] == [3.99, 2.82]
    assert info['intensity'] == [100.0, 50.0]
    assert len(info['two_theta']) == 2


def test_xml_to_info_mismatched_da_intensity_counts_raises(tmp_path):
    xml_path = _write_xml(tmp_path, da_values=('3.99', '2.82'), intensity_values=('100',))
    with pytest.raises(ValueError, match='mismatch'):
        extraction.xml_to_info(xml_path)


def test_xml_to_info_drops_unreachable_reflections(tmp_path):
    # d = 0.5 A is below lambda/2 for CuKa (0.771 A), so its 2theta is NaN
    # (sin(theta) > 1). That reflection -- and its paired intensity -- must be
    # dropped so no NaN reaches the reference pattern; the remaining reflections
    # stay aligned across d / two_theta / intensity.
    xml_path = _write_xml(
        tmp_path,
        da_values=('3.99', '0.5', '2.82'),
        intensity_values=('100', '77', '50'),
    )
    info = extraction.xml_to_info(xml_path)

    assert info['d'] == [3.99, 2.82]
    assert info['intensity'] == [100.0, 50.0]
    assert len(info['two_theta']) == 2
    assert all(np.isfinite(tt) for tt in info['two_theta'])


def test_xml_to_info_pairs_da_and_intensity_per_reflection(tmp_path):
    # Regression: d-spacing and intensity must be read from the SAME reflection
    # node. Each reflection carries extra siblings (theta, h/k/l) and the
    # container <intensity> is blank -- the exact shape of real ICDD cards --
    # so a parser that scanned all <da> and all <intensity> independently could
    # drop the blank containers unevenly and silently shift intensities against
    # their 2theta. Per-reflection reading keeps them aligned.
    reflections = (
        '<intensity>\n<theta>25.0</theta><da>4.00</da><intensity>100</intensity>'
        '<h>1</h><k>0</k><l>0</l></intensity>'
        '<intensity>\n<theta>31.0</theta><da>2.90</da><intensity>40</intensity>'
        '<h>1</h><k>1</k><l>0</l></intensity>'
        '<intensity>\n<theta>45.0</theta><da>2.00</da><intensity>75</intensity>'
        '<h>2</h><k>0</k><l>0</l></intensity>'
    )
    xml_text = f"""<root>
        <empirical_formula>NaCl</empirical_formula>
        <z>4</z>
        <spgr>F m -3 m (225)</spgr>
        <pdf_number>00-001-2345</pdf_number>
        <cell_vol>175.616</cell_vol>
        <xstal_system>cubic</xstal_system>
        <cell_a>5.6</cell_a>
        <graphs><stick_series>{reflections}</stick_series></graphs>
    </root>"""
    xml_path = tmp_path / 'ref.xml'
    xml_path.write_text(xml_text)

    info = extraction.xml_to_info(str(xml_path))

    assert info['d'] == [4.00, 2.90, 2.00]
    assert info['intensity'] == [100.0, 40.0, 75.0]


def test_xml_to_info_collapses_overlapped_reflections(tmp_path):
    # ICDD writes an accidentally degenerate group (here 110/101, as the
    # pseudo-cubic cards do) as one line per contributing hkl, each repeating
    # the same <da> and the same group intensity with an 'm' overlap flag. The
    # listed value is already the group total, so the repeats must be dropped,
    # not kept: two identical sticks at one 2theta would be summed into a
    # doubled peak by the Gaussian broadening downstream.
    reflections = (
        '<intensity>\n<theta>25.0</theta><da>4.00</da><intensity>100</intensity>'
        '<h>1</h><k>0</k><l>0</l></intensity>'
        '<intensity>\n<theta>31.0</theta><da>2.90</da><intensity>40.0m</intensity>'
        '<h>1</h><k>1</k><l>0</l></intensity>'
        '<intensity>\n<theta>31.0</theta><da>2.90</da><intensity>40.0m</intensity>'
        '<h>1</h><k>0</k><l>1</l></intensity>'
        '<intensity>\n<theta>45.0</theta><da>2.00</da><intensity>75</intensity>'
        '<h>2</h><k>0</k><l>0</l></intensity>'
    )
    xml_text = f"""<root>
        <empirical_formula>NaCl</empirical_formula>
        <z>4</z>
        <spgr>F m -3 m (225)</spgr>
        <pdf_number>00-001-2345</pdf_number>
        <cell_vol>175.616</cell_vol>
        <xstal_system>cubic</xstal_system>
        <cell_a>5.6</cell_a>
        <graphs><stick_series>{reflections}</stick_series></graphs>
    </root>"""
    xml_path = tmp_path / 'ref.xml'
    xml_path.write_text(xml_text)

    info = extraction.xml_to_info(str(xml_path))

    assert info['d'] == [4.00, 2.90, 2.00]
    assert info['intensity'] == [100.0, 40.0, 75.0]
    assert len(info['two_theta']) == 3


# =============================================================================
# cif_to_info
# =============================================================================
def test_cif_to_info_parses_nacl_cif(tmp_path):
    lattice = Lattice.cubic(5.6)
    structure = Structure.from_spacegroup(
        'Fm-3m', lattice, ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    cif_path = tmp_path / 'nacl.cif'
    structure.to(filename=str(cif_path))

    info = extraction.cif_to_info(str(cif_path))

    assert info['chemical_formula'] == 'NaCl'
    assert info['spg_num'] == 225
    assert info['z'] == 4
    assert len(info['two_theta']) > 0
    assert len(info['two_theta']) == len(info['intensity'])
