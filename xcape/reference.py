"""Reference loading and preprocessing utilities for XCAPE."""

import logging
from pathlib import Path
from typing import Any

from .modules.distance_constraint import get_distance_constraint
from .modules.extraction import cif_to_info, xml_to_info
from .utils.types import ReferenceInfo
from .utils.core_utils import update_general_info
from .validation import validate_reference_info

logger = logging.getLogger(__name__)


def prepare_reference(config: dict[str, Any]) -> ReferenceInfo:
    """Prepare the reference XRD pattern information from XML or CIF input."""
    input_setting = config.get('input_setting', {})
    xrd_setting = config.get('XRD_setting', {})
    xml_path = input_setting.get('xml_path')
    cif_path = input_setting.get('cif_path')
    wavelength = xrd_setting.get('wave_length')
    two_theta_range = xrd_setting.get('two_theta_range')
    two_theta_tol = float(xrd_setting.get('two_theta_tol', 0.0))

    if xml_path and cif_path:
        raise ValueError('Do not set both "xml_path" and "cif_path". Choose one')
    if not wavelength:
        raise ValueError('"XRD_setting.wave_length" must be defined.')
    if xml_path:
        target_path = _resolve_input_file_path(xml_path, '*.xml', 'XML')
        ref_info = xml_to_info(target_path, wavelength=wavelength)
    elif cif_path:
        target_path = _resolve_input_file_path(cif_path, '*.cif', 'CIF')
        cif_two_theta_range = None
        if two_theta_range is not None:
            cif_two_theta_range = _expand_two_theta_range(two_theta_range, two_theta_tol)
        ref_info = cif_to_info(
            target_path,
            wavelength=wavelength,
            two_theta_range=cif_two_theta_range,
        )
    else:
        raise ValueError('Either "xml_path" or "cif_path" settings must be defined.')

    ref_info = _apply_two_theta_range(
        ref_info,
        _expand_two_theta_range(two_theta_range, two_theta_tol),
    )
    validate_reference_info(ref_info)
    update_general_info(config, ref_info)
    return ref_info


def ensure_distance_constraint(config: dict[str, Any], calc) -> Any:
    """Ensure distance constraints exist and convert them into a PyXtal tolerance matrix."""
    if 'distance_constraint' not in config:
        logger.info("No distance constraints provided. Generating distance constraints using MQA MD...")
        dist_constraint = get_distance_constraint(config, calc)
        config['distance_constraint'] = dist_constraint
    else:
        logger.info("Using user-provided distance constraints.")
        dist_constraint = config['distance_constraint']

    # This path currently supports the pyxtal backend only.
    from pyxtal.tolerance import Tol_matrix

    tol_matrix = Tol_matrix()

    for elem, dist in dist_constraint.items():
        e1, e2 = elem.split('-')
        tol_matrix.set_tol(e1, e2, dist)

    return tol_matrix


def _resolve_input_file_path(input_path_str: str, pattern: str, extension_label: str) -> str:
    """Resolve an input file path, preferring direct file paths over directories."""
    input_path = Path(input_path_str)
    if input_path.is_file():
        return str(input_path)
    if not input_path.is_dir():
        raise ValueError(f'Input path does not exist: {input_path_str}')

    logger.info(
        'Directory input detected for %s reference: %s. '
        'A direct file path is recommended for clearer, reproducible runs.',
        extension_label,
        input_path_str,
    )
    input_files = sorted(path for path in input_path.glob(pattern) if path.is_file())
    if not input_files:
        raise ValueError(
            f'No {extension_label} files matching "{pattern}" found in directory input: {input_path_str}'
        )
    if len(input_files) > 1:
        matched_files = ', '.join(path.name for path in input_files)
        raise ValueError(
            f'Directory input for {extension_label} must contain exactly one file matching "{pattern}", '
            f'but found {len(input_files)} in {input_path_str}: {matched_files}'
        )
    return str(input_files[0])


def _apply_two_theta_range(
    ref_info: ReferenceInfo,
    two_theta_range: Any,
) -> ReferenceInfo:
    """Filter extracted reference peaks to the configured 2theta range."""
    if two_theta_range is None:
        return ref_info

    range_min = float(two_theta_range[0])
    range_max = float(two_theta_range[1])
    keep = [
        range_min <= float(two_theta) <= range_max
        for two_theta in ref_info['two_theta']
    ]
    if not any(keep):
        raise ValueError(
            'No reference peaks remain after applying "XRD_setting.two_theta_range". '
            f'Configured range: {two_theta_range}'
        )

    ref_info['two_theta'] = [
        float(v) for v, k in zip(ref_info['two_theta'], keep) if k
    ]
    ref_info['intensity'] = [
        float(v) for v, k in zip(ref_info['intensity'], keep) if k
    ]
    # Keep the d-spacing list (present on the XML path) index-aligned with the
    # cropped two_theta/intensity so downstream consumers can't see a stale 'd'.
    if 'd' in ref_info:
        ref_info['d'] = [v for v, k in zip(ref_info['d'], keep) if k]
    return ref_info


def _expand_two_theta_range(
    two_theta_range: Any,
    two_theta_tol: float,
) -> tuple[float, float] | None:
    """Expand a configured 2theta range by the matching tolerance."""
    if two_theta_range is None:
        return None
    return (
        float(two_theta_range[0]) - two_theta_tol,
        float(two_theta_range[1]) + two_theta_tol,
    )
