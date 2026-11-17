"""Reference data extraction module for XCAPE.

This module provides functions to extract crystallographic information
from CIF and XML files for use as reference data in crystal structure prediction.
"""
import logging
import re
from collections import defaultdict
from typing import Dict, Any, Union, List, Optional, Set, Tuple
from pathlib import Path

import numpy as np
import math
import xml.etree.ElementTree as ET

from pyxtal import pyxtal

from pymatgen.core import Structure, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from ..utils.core_utils import get_volume_from_lattice, get_Z_from_density
from ..utils.xrd_utils import d_to_two_theta

logger = logging.getLogger(__name__)


def cif_to_info(
    cif_path: Union[str, Path],
    wavelength: str = 'CuKa',
    two_theta_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Parse a CIF file and return reference information dictionary.
    
    Args:
        cif_path: Path to the CIF file.
        wavelength: X-ray wavelength for XRD calculation (default: 'CuKa').
        two_theta_range: Optional 2theta range passed to pymatgen XRDCalculator.
    
    Returns:
        Dictionary containing material composition, lattice parameters,
        space group number, and XRD pattern data.
    Raises:
        FileNotFoundError: If the CIF file does not exist.
    """
    cif_path = str(cif_path)
    
    structure = Structure.from_file(cif_path)

    xtal = pyxtal()
    xtal.from_seed(structure)

    a, b, c, alpha, beta, gamma = xtal.lattice.get_para()
    a, b, c = float(a), float(b), float(c)
    alpha, beta, gamma = math.degrees(alpha), math.degrees(beta), math.degrees(gamma)

    lat = [a, b, c, alpha, beta, gamma]
    lat = [round(x, 5) for x in lat]
    
    volume = get_volume_from_lattice(lat)

    pretty_formula = structure.composition.reduced_formula
    Z = get_Z_from_density(pretty_formula, structure.density, volume)

    comp_dict = Composition(pretty_formula).get_el_amt_dict()
    comp_dict = {k: int(v) * Z for k, v in comp_dict.items()}
    
    sga = SpacegroupAnalyzer(structure)
    spg_num = sga.get_space_group_number()

    xrd_calc = XRDCalculator(wavelength=wavelength)
    pattern = xrd_calc.get_pattern(structure, two_theta_range=two_theta_range)
    two_theta = pattern.x
    intensity = pattern.y

    info = {
        'material': comp_dict,
        'pdf_number': None,  # No ICDD PDF number for a CIF source.
        'source_path': str(cif_path),
        'chemical_formula': pretty_formula,
        'z': Z,
        'volume': volume,
        'lattice_parameters': lat,
        'spg_symbol': sga.get_space_group_symbol(),
        'spg_num': spg_num,
        'two_theta': two_theta,
        'intensity': intensity,
    }

    return info


def find_text_by_tag(root: ET.Element, tag_name: str) -> List[str]:
    """Find all text contents for elements with the specified XML tag.
    
    Args:
        root: The root element of the XML tree.
        tag_name: The name of the tag to search for.
    
    Returns:
        A list of text contents from all matching elements.
    """
    elements = root.findall(f".//{tag_name}")
    return [elem.text for elem in elements if elem.text is not None]


def _get_required_text(root: ET.Element, tag_name: str) -> str:
    """Return the first text value for a required XML tag.

    Raises:
        ValueError: If the tag is missing or contains no text.
    """
    values = find_text_by_tag(root, tag_name)
    if not values or values[0] is None:
        raise ValueError(f"Required XML tag '{tag_name}' is missing or empty")
    return values[0]


def _parse_leading_float(text: Optional[str]) -> Optional[float]:
    """Parse the first numeric value in an XML text node.

    Handles the value formats that appear in ICDD PDF cards:
    - parenthetical uncertainties are dropped, e.g. "5.839(2)" -> 5.839,
      "33(5)" -> 33.0 (each group removed individually, so multiple groups
      are handled correctly rather than greedily);
    - whitespace-only / empty nodes (e.g. the "\\n" separators interleaved
      in ``<intensity>`` lists) yield None so callers can skip them.

    Returns None when the node contains no number; separate numbers are never
    concatenated together.
    """
    if text is None:
        return None
    # Remove each parenthetical group individually (non-greedy per group).
    cleaned = re.sub(r'\([^)]*\)', '', text)
    match = re.search(r'[-+]?\d*\.?\d+', cleaned)
    if match is None:
        return None
    return float(match.group())


def _get_required_float(root: ET.Element, tag_name: str) -> float:
    """Return the first text value for a required XML tag as float.

    Parenthetical uncertainties are stripped, e.g. "10.0(2)" -> 10.0.
    """
    text = _get_required_text(root, tag_name)
    value = _parse_leading_float(text)
    if value is None:
        raise ValueError(f"Required XML tag '{tag_name}' is not numeric: {text!r}")
    return value


def _get_optional_float(root: ET.Element, tag_name: str) -> Optional[float]:
    """Return the first text value for an optional XML tag as float."""
    try:
        return _get_required_float(root, tag_name)
    except ValueError:
        return None


def _extract_real_cell_parameters(root: ET.Element) -> List[Optional[float]]:
    """Extract raw unit-cell parameters in [a, b, c, alpha, beta, gamma] order."""
    cell_tags = (
        'cell_a',
        'cell_b',
        'cell_c',
        'cell_alpha',
        'cell_beta',
        'cell_gamma',
    )
    return [_get_optional_float(root, tag_name) for tag_name in cell_tags]


def _normalize_lattice_parameters(
    crystal_system: str,
    real_cell_parameters: List[Optional[float]],
) -> List[Optional[float]]:
    """Fill symmetry-constrained lattice parameters from raw cell values."""
    a, b, c, alpha, beta, gamma = real_cell_parameters

    system = crystal_system.lower()

    # Trigonal structures use either rhombohedral (alpha != 90) or hexagonal setting.
    if system in ('trigonal', 'rhombohedral'):
        if alpha is not None and (c is None or not math.isclose(alpha, 90.0, abs_tol=1.0)):
            return [a, a, a, alpha, alpha, alpha]
        return [a, a, c, 90.0, 90.0, 120.0]

    lattice_by_system = {
        'cubic': [a, a, a, 90.0, 90.0, 90.0],
        'tetragonal': [a, a, c, 90.0, 90.0, 90.0],
        'orthorhombic': [a, b, c, 90.0, 90.0, 90.0],
        'hexagonal': [a, a, c, 90.0, 90.0, 120.0],
        'monoclinic': [a, b, c, 90.0, beta, 90.0],
    }
    return lattice_by_system.get(system, real_cell_parameters)


def xml_to_info(xml_path: str, wavelength: str = 'CuKa') -> Dict[str, Any]:
    """Parse an XML file and return reference information dictionary.
    
    Args:
        xml_path: Path to the XML file (typically from ICDD PDF database).
        wavelength: X-ray wavelength for 2θ calculation (default: 'CuKa').
    
    Returns:
        Dictionary containing material composition, lattice parameters,
        space group number, d-spacings, and XRD pattern data.
    
    Raises:
        FileNotFoundError: If the XML file does not exist.
        ET.ParseError: If the XML file is malformed.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract basic crystallographic data
    formula_raw = _get_required_text(root, 'empirical_formula')
    formula = ''.join(formula_raw.split())
    counts_per_fu = parse_formula_counts(formula)

    Z = _get_required_float(root, 'z')
    counts_per_cell = {el: round(cnt * Z) for el, cnt in counts_per_fu.items()}

    # Extract space group symbol and number from format like "P -1 (2)"
    spgr_text = _get_required_text(root, 'spgr')
    spg_symbol = re.sub(r'\s*\(\s*\d+\s*\)\s*$', '', spgr_text).strip()
    spg_match = re.search(r'\(\s*(\d+)\s*\)\s*$', spgr_text)
    if spg_match is None:
        raise ValueError(
            f"Could not parse space group number from 'spgr' text: {spgr_text!r}"
        )
    spg_num = int(spg_match.group(1))

    # Extract (d-spacing, intensity) per reflection so the two columns are read
    # from the same node and cannot become mis-paired (see _extract_reflections).
    reflections = _extract_reflections(root)
    missing_intensity = sum(1 for _, intensity in reflections if intensity is None)
    if missing_intensity:
        raise ValueError(
            f"XRD data mismatch in '{xml_path}': "
            f"{missing_intensity} reflection(s) have a d-spacing but no parseable intensity"
        )

    d_spacings = [d for d, _ in reflections]
    intensity_values = [intensity for _, intensity in reflections]
    two_theta_values = np.round(
        d_to_two_theta(d_spacings, wavelength=wavelength), 3
    ).tolist()

    pdf_number = _get_required_text(root, 'pdf_number')
    cell_volume = _get_required_float(root, 'cell_vol')
    crystal_system = _get_required_text(root, 'xstal_system')
    real_cell_parameters = _extract_real_cell_parameters(root)
    lattice_parameters = _normalize_lattice_parameters(
        crystal_system,
        real_cell_parameters,
    )

    # Verify the reconstructed cell against the card's own 'cell_vol'. A misidentified
    # symmetry setting or a missing cell_* tag surfaces here instead of leaking a None
    # into downstream consumers such as get_ltype().
    if None in lattice_parameters:
        raise ValueError(
            f"Incomplete lattice parameters for crystal system '{crystal_system}' "
            f"in '{xml_path}': {lattice_parameters}"
        )
    reconstructed_volume = get_volume_from_lattice(lattice_parameters)
    if abs(reconstructed_volume - cell_volume) > 0.01 * cell_volume:
        raise ValueError(
            f"Reconstructed cell volume {reconstructed_volume:.2f} disagrees with "
            f"'cell_vol' {cell_volume:.2f} in '{xml_path}': the symmetry setting for "
            f"'{crystal_system}' is likely misidentified ({lattice_parameters})"
        )

    # Drop reflections whose 2theta is non-finite. d_to_two_theta emits NaN for
    # reflections unreachable at this wavelength (d < lambda/2, i.e. sin(theta) > 1).
    # A NaN 2theta would poison the Gaussian-broadened Wasserstein pattern (one
    # NaN makes the whole curve NaN) and the default 2theta range (min/max over a
    # list containing NaN is NaN). r_factor_peak_pairing already skips non-finite
    # peaks, so removing them here keeps every downstream metric consistent.
    # d_spacings, two_theta_values, and intensity_values are index-aligned here.
    finite_flags = [math.isfinite(tt) for tt in two_theta_values]
    if not all(finite_flags):
        dropped = finite_flags.count(False)
        logger.warning(
            "Dropping %d reference reflection(s) with non-finite 2theta "
            "(d-spacing below lambda/2, unreachable at wavelength %s) from '%s'.",
            dropped, wavelength, xml_path,
        )
        d_spacings = [d for d, keep in zip(d_spacings, finite_flags) if keep]
        two_theta_values = [tt for tt, keep in zip(two_theta_values, finite_flags) if keep]
        intensity_values = [iv for iv, keep in zip(intensity_values, finite_flags) if keep]

    info = {
        'material': counts_per_cell,
        'pdf_number': pdf_number,
        'chemical_formula': formula,
        'z': Z,
        'volume': cell_volume,
        'lattice_parameters': lattice_parameters,  # a, b, c, alpha, beta, gamma
        'spg_symbol': spg_symbol,
        'spg_num': spg_num,
        'd': d_spacings,
        'two_theta': two_theta_values,
        'intensity': intensity_values,
    }

    return info


def _extract_reflections(root: ET.Element) -> List[Tuple[float, Optional[float]]]:
    """Extract ``(d_spacing, intensity)`` pairs one reflection at a time.

    Real ICDD PDF cards nest each reflection under ``stick_series`` as a
    container element that holds its own ``<da>`` and (inner) ``<intensity>``
    together, e.g.::

        <stick_series>
          <intensity>            <!-- reflection container, blank text -->
            <theta>24.710</theta>
            <da>3.600</da>
            <intensity>15</intensity>   <!-- the actual intensity value -->
            ...
          </intensity>
          ...
        </stick_series>

    A reflection node is identified by having a direct ``<da>`` child; its
    d-spacing and intensity are read from that *same* node. This guarantees
    the two stay paired (parsing the two columns as independent global scans
    can silently mis-align an intensity with the wrong 2theta).

    Overlapped reflections are collapsed to one entry. ICDD writes an
    accidentally degenerate group (e.g. the 112/211 pair of the pseudo-cubic
    PDF 04-013-8738) as one line per contributing hkl, each repeating the same
    ``<da>`` and the same intensity with an ``m`` flag appended to the intensity
    text. That listed value is the *group* total, so the extra lines must be
    dropped rather than summed; keeping them would place identical sticks at a
    single 2theta, which the Gaussian broadening downstream adds into a doubled
    peak.

    Returns:
        A list of ``(d, intensity)`` tuples in document order, at most one per
        distinct d-spacing. ``intensity`` is ``None`` when the reflection has no
        parseable intensity value (the caller decides how to treat that).
    """
    reflections: List[Tuple[float, Optional[float]]] = []
    seen_d: Set[float] = set()
    duplicates = 0
    for node in root.iter():
        da_node = node.find('da')
        if da_node is None:
            continue
        d = _parse_leading_float(da_node.text)
        if d is None:
            continue
        if d in seen_d:
            duplicates += 1
            continue
        seen_d.add(d)
        intensity_node = node.find('intensity')
        intensity = (
            _parse_leading_float(intensity_node.text)
            if intensity_node is not None
            else None
        )
        reflections.append((d, intensity))
    if duplicates:
        logger.info(
            "Collapsed %d overlapped reference reflection(s) that repeat an "
            "earlier d-spacing; %d unique reflection(s) kept.",
            duplicates, len(reflections),
        )
    return reflections


def _parse_intensities(raw_intensities: List[str]) -> List[float]:
    """Parse intensity values from XML text content.
    
    Args:
        raw_intensities: List of raw intensity strings from XML.
    
    Returns:
        List of parsed intensity values as floats.
    """
    parsed = []
    for raw_value in raw_intensities:
        value = _parse_leading_float(raw_value)
        if value is not None:
            parsed.append(value)
    return parsed


def parse_formula_counts(formula: str) -> Dict[str, float]:
    """Parse chemical formula string into element counts.
    
    Handles various formula formats including those with separators
    like commas, middle dots (·), asterisks, and parentheses.
    
    Args:
        formula: Chemical formula string (e.g., 'Li2O', 'Ca3(PO4)2').
    
    Returns:
        Dictionary mapping element symbols to their counts.
    
    Examples:
        >>> parse_formula_counts('Li2O')
        {'Li': 2.0, 'O': 1.0}
        >>> parse_formula_counts('H2O')
        {'H': 2.0, 'O': 1.0}
    """
    normalized = formula.replace(",", " ").replace("·", " ").replace("*", " ")
    total_counts = defaultdict(float)

    def parse_number(text: str, start: int) -> tuple[float, int]:
        match = re.match(r"([0-9]*\.?[0-9]*)", text[start:])
        if match is None:
            return 1.0, start
        number_text = match.group(1)
        # A bare "." matches the pattern but is not a valid float; only parse
        # when at least one digit is present, otherwise treat as "no number"
        # (implicit multiplier 1.0) and do not consume the character.
        if not any(ch.isdigit() for ch in number_text):
            return 1.0, start
        return float(number_text), start + len(number_text)

    def parse_segment(text: str, start: int = 0) -> tuple[Dict[str, float], int]:
        counts = defaultdict(float)
        idx = start

        while idx < len(text):
            char = text[idx]

            if char == '(':
                inner_counts, idx = parse_segment(text, idx + 1)
                multiplier, idx = parse_number(text, idx)
                for element, count in inner_counts.items():
                    counts[element] += count * multiplier
                continue

            if char == ')':
                return dict(counts), idx + 1

            if char.isupper():
                end = idx + 1
                while end < len(text) and text[end].islower():
                    end += 1
                element = text[idx:end]
                multiplier, idx = parse_number(text, end)
                counts[element] += multiplier
                continue

            idx += 1

        return dict(counts), idx

    for token in normalized.split():
        token_multiplier, start_idx = parse_number(token, 0)
        token_counts, _ = parse_segment(token, start_idx)
        for element, count in token_counts.items():
            total_counts[element] += count * token_multiplier

    return dict(total_counts)
