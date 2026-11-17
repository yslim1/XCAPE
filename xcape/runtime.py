"""Runtime setup utilities for XCAPE."""

import logging
from pathlib import Path
from typing import Any

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.structure_matcher import StructureMatcher

from .mlips.sevennet import load_sevennet_calc
from .modules.generation import random_generation_pyxtal
from .utils.types import Environment

# Default StructureMatcher tolerances
DEFAULT_LTOL = 0.2
DEFAULT_STOL = 0.3
DEFAULT_ANGLE_TOL = 5.0

# Log file names
XCAPE_LOG_FILENAME = 'xcape.log'
LOG_FILENAME = 'generation.log'
MEMORY_LOG_FILENAME = 'memory.log'
NO_SYMMETRY_LOG_FILENAME = 'non-sym.log'

# Log header format
LOG_HEADER = (
    f"{'Gen':>4} {'Pop':>4} {'Type':>15} {'Parent':>8} "
    f"{'E':>8} {'RF':>8} {'WD':>8} {'Gen_t':>6} "
    f"{'Rel_t':>6} {'Status'}\n"
)


def setup_logging(output_path: Path) -> None:
    """Set up XCAPE logging for both file and console output."""
    log_path = output_path / XCAPE_LOG_FILENAME

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )


def setup_environment(config: dict[str, Any]) -> Environment:
    """Build the runtime environment used by the XCAPE main loop."""
    output_path = Path(config['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    wavelength = config['XRD_setting']['wave_length']

    return Environment(
        output_path=output_path,
        generator=random_generation_pyxtal,
        mlip=load_sevennet_calc(config),
        xrd_calc=XRDCalculator(wavelength=wavelength),
        structure_matcher=build_structure_matcher(config),
    )


def build_structure_matcher(config: dict[str, Any]) -> StructureMatcher:
    """Build the StructureMatcher used for deduplication."""
    matcher_cfg = config.get('structure', {}).get('dedup', {}).get('structure_matcher', {})
    if not matcher_cfg:
        return StructureMatcher(
            ltol=DEFAULT_LTOL,
            stol=DEFAULT_STOL,
            angle_tol=DEFAULT_ANGLE_TOL,
            primitive_cell=False,
        )

    return StructureMatcher(
        ltol=float(matcher_cfg.get('ltol', DEFAULT_LTOL)),
        stol=float(matcher_cfg.get('stol', DEFAULT_STOL)),
        angle_tol=float(matcher_cfg.get('angle_tol', DEFAULT_ANGLE_TOL)),
        primitive_cell=bool(matcher_cfg.get('primitive_cell', False)),
    )


def initialize_generation_log(log_file: Path) -> None:
    """Write the tabular generation log header when the file is empty."""
    if log_file.exists() and log_file.stat().st_size > 0:
        return

    with open(log_file, 'a') as stream:
        stream.write(LOG_HEADER)
