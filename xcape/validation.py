"""Validation helpers for XCAPE configuration and reference metadata."""

from typing import Any

from .utils.types import ReferenceInfo
from .utils.xrd_utils import WAVELENGTHS

DEFAULT_EARLY_STOPPING_THRESHOLD = 5 * 1e-3
DEFAULT_ADD_NO_SYMMETRY_RELAX_RATIO = 0.3
ALLOWED_XRD_NORMALIZE = {None, 'max', 'sum', 'scale'}
# The seven mutation-move ratio keys. When `mutation_ratios` is provided, all
# seven must be present and sum to 1.0 (checked below). This is a key-set only,
# not a set of default values, so the ratios each user supplies are validated
# against these names rather than defaulted from them.
REQUIRED_MUTATION_RATIO_KEYS = frozenset({
    'swap_same_mult_ratio',
    'split_site_ratio',
    'merge_site_ratio',
    'swap_diff_elem_ratio',
    'site_reconfig_ratio',
    'resample_pos_ratio',
    'multi_resample_pos_ratio',
})

# The three keep_max mixing keys. When `mix_ratios` is provided, all three must
# be present and sum to 1.0 (checked below). Partial dicts are rejected because
# the missing keys would otherwise be filled from the DEFAULT_KEEP_MAX_* ratios
# in modules.postprocess, pushing the total above 1.0.
REQUIRED_MIX_RATIO_KEYS = frozenset({
    'r_factor',
    'energy',
    'wasserstein',
})


def validate_config(config: dict[str, Any]) -> None:
    """Validate critical configuration values before starting the search."""
    output_path = config.get('output_path')
    input_cfg = config.get('input_setting', {})
    structure_cfg = config.get('structure', {})
    optimizer_cfg = config.get('optimizer', {})
    xrd_cfg = config.get('XRD_setting', {})

    xml_path = input_cfg.get('xml_path')
    cif_path = input_cfg.get('cif_path')
    generation = int(structure_cfg.get('generation', 0))
    population = int(structure_cfg.get('population', 0))
    _raw_random_ratio = structure_cfg.get('random_ratio')
    weighted_parent_ratio = float(
        structure_cfg.get('mutation', {}).get('selection_method', {}).get('weighted_parent_ratio', 0.8)
    )
    mutation_ratios = structure_cfg.get('mutation', {}).get('mutation_ratios', {})
    early_stopping_threshold = float(
        structure_cfg.get('early_stopping_threshold', DEFAULT_EARLY_STOPPING_THRESHOLD)
    )
    add_no_symmetry_relax = structure_cfg.get('add_no_symmetry_relax', False)
    add_no_symmetry_relax_ratio = structure_cfg.get(
        'add_no_symmetry_relax_ratio', DEFAULT_ADD_NO_SYMMETRY_RELAX_RATIO
    )
    keep_max = structure_cfg.get('dedup', {}).get('keep_max')
    optimizer_type = optimizer_cfg.get('type')
    fix_symmetry = optimizer_cfg.get('fix_symmetry', False)
    wavelength = xrd_cfg.get('wave_length')
    two_theta_tol = float(xrd_cfg.get('two_theta_tol', 0.0))
    two_theta_range = xrd_cfg.get('two_theta_range')
    normalize = xrd_cfg.get('normalize')
    if isinstance(normalize, str):
        normalize = normalize.lower()

    if not output_path:
        raise ValueError('"output_path" must be defined.')
    seed = config.get('seed')
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f'"seed" must be an integer when set, got: {seed!r}')
        if seed < 0:
            raise ValueError(f'"seed" must be non-negative when set, got: {seed}')
    if bool(xml_path) == bool(cif_path):
        raise ValueError('Exactly one of "input_setting.xml_path" or "input_setting.cif_path" must be set.')
    if xml_path is not None and not isinstance(xml_path, str):
        raise ValueError('"input_setting.xml_path" must be a string path when set.')
    if cif_path is not None and not isinstance(cif_path, str):
        raise ValueError('"input_setting.cif_path" must be a string path when set.')
    if generation <= 0:
        raise ValueError(f'"structure.generation" must be a positive integer, got: {generation}')
    if population <= 0:
        raise ValueError(f'"structure.population" must be a positive integer, got: {population}')
    if _raw_random_ratio is None:
        raise ValueError('"structure.random_ratio" must be defined.')
    random_ratio = float(_raw_random_ratio)
    if not 0.0 <= random_ratio <= 1.0:
        raise ValueError(f'"structure.random_ratio" must be between 0 and 1, got: {random_ratio}')
    if not 0.0 <= weighted_parent_ratio <= 1.0:
        raise ValueError(
            '"structure.mutation.selection_method.weighted_parent_ratio" must be between 0 and 1, '
            f'got: {weighted_parent_ratio}'
        )
    if mutation_ratios:
        missing_ratio_keys = sorted(REQUIRED_MUTATION_RATIO_KEYS - set(mutation_ratios))
        if missing_ratio_keys:
            raise ValueError(
                '"structure.mutation.mutation_ratios" must define all mutation ratio keys when set. '
                f'Missing: {missing_ratio_keys}'
            )
        try:
            mutation_ratio_sum = sum(float(mutation_ratios[key]) for key in REQUIRED_MUTATION_RATIO_KEYS)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                '"structure.mutation.mutation_ratios" values must be numeric.'
            ) from exc
        if abs(mutation_ratio_sum - 1.0) > 1e-6:
            raise ValueError(
                '"structure.mutation.mutation_ratios" must sum to 1.0, '
                f'got: {mutation_ratio_sum}'
            )
    if early_stopping_threshold < 0.0:
        raise ValueError(
            '"structure.early_stopping_threshold" must be non-negative, '
            f'got: {early_stopping_threshold}'
        )
    if not isinstance(add_no_symmetry_relax, bool):
        raise ValueError(
            '"structure.add_no_symmetry_relax" must be a boolean, '
            f'got: {add_no_symmetry_relax!r}'
        )
    try:
        add_no_symmetry_relax_ratio = float(add_no_symmetry_relax_ratio)
    except (TypeError, ValueError):
        raise ValueError(
            '"structure.add_no_symmetry_relax_ratio" must be a number, '
            f'got: {add_no_symmetry_relax_ratio!r}'
        )
    if not 0.0 <= add_no_symmetry_relax_ratio <= 1.0:
        raise ValueError(
            '"structure.add_no_symmetry_relax_ratio" must be between 0 and 1, '
            f'got: {add_no_symmetry_relax_ratio}'
        )
    if optimizer_type not in {'sequential', 'batch'}:
        raise ValueError(
            '"optimizer.type" must be either "sequential" or "batch", '
            f'got: {optimizer_type}'
        )
    if not isinstance(fix_symmetry, bool):
        raise ValueError(f'"optimizer.fix_symmetry" must be a boolean, got: {fix_symmetry}')
    if wavelength not in WAVELENGTHS:
        raise ValueError(
            f'"XRD_setting.wave_length" got {wavelength!r}. '
            f'Must be one of: {sorted(WAVELENGTHS)}'
        )
    if two_theta_tol < 0.0:
        raise ValueError(f'"XRD_setting.two_theta_tol" must be non-negative, got: {two_theta_tol}')
    if two_theta_range is not None:
        if not isinstance(two_theta_range, (list, tuple)) or len(two_theta_range) != 2:
            raise ValueError(
                '"XRD_setting.two_theta_range" must be a list or tuple with exactly two values.'
            )
        try:
            range_min = float(two_theta_range[0])
            range_max = float(two_theta_range[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                '"XRD_setting.two_theta_range" values must be numeric.'
            ) from exc
        if range_min >= range_max:
            raise ValueError(
                '"XRD_setting.two_theta_range" must satisfy min < max, '
                f'got: {two_theta_range}'
            )
    if normalize not in ALLOWED_XRD_NORMALIZE:
        raise ValueError(
            '"XRD_setting.normalize" must be one of null, "max", "sum", or "scale", '
            f'got: {normalize}'
        )
    if keep_max is not None:
        try:
            keep_max_int = int(keep_max)
        except (TypeError, ValueError):
            raise ValueError(f'"structure.dedup.keep_max" must be an integer, got: {keep_max!r}')
        if keep_max_int <= 0:
            raise ValueError(f'"structure.dedup.keep_max" must be positive when set, got: {keep_max}')
    dedup_cfg = structure_cfg.get('dedup', {})
    if 'wd' in dedup_cfg and 'wasserstein' in dedup_cfg:
        raise ValueError(
            '"structure.dedup" should set only one of "wasserstein" or legacy "wd".'
        )
    mix_ratios = dedup_cfg.get('mix_ratios')
    if mix_ratios is not None and not isinstance(mix_ratios, dict):
        raise ValueError(
            f'"structure.dedup.mix_ratios" must be a mapping, got: {mix_ratios!r}'
        )
    if mix_ratios:
        missing_mix_ratio_keys = sorted(REQUIRED_MIX_RATIO_KEYS - set(mix_ratios))
        if missing_mix_ratio_keys:
            raise ValueError(
                '"structure.dedup.mix_ratios" must define all mix ratio keys when set. '
                f'Missing: {missing_mix_ratio_keys}'
            )
        total = 0.0
        for key in sorted(REQUIRED_MIX_RATIO_KEYS):
            try:
                val = float(mix_ratios[key])
            except (TypeError, ValueError):
                raise ValueError(
                    f'"structure.dedup.mix_ratios.{key}" must be a number, '
                    f'got: {mix_ratios[key]!r}'
                )
            if val < 0.0:
                raise ValueError(
                    f'"structure.dedup.mix_ratios.{key}" must be non-negative, got: {val}'
                )
            total += val
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f'"structure.dedup.mix_ratios" entries must sum to 1.0, got: {total:.4f}'
            )


def validate_reference_info(ref_info: ReferenceInfo) -> None:
    """Validate the minimum reference fields required by downstream modules."""
    required_keys = [
        'chemical_formula',
        'lattice_parameters',
        'z',
        'spg_num',
        'spg_symbol',
        'two_theta',
        'intensity',
    ]
    for key in required_keys:
        if key not in ref_info:
            raise ValueError(f'Reference info is missing required key: {key}')
