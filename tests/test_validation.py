import pytest

from xcape.validation import validate_config, validate_reference_info


def test_sample_config_is_valid(sample_config):
    validate_config(sample_config)


def test_missing_output_path_raises(config_factory):
    config = config_factory()
    del config['output_path']
    with pytest.raises(ValueError, match='output_path'):
        validate_config(config)


def test_both_xml_and_cif_path_raises(config_factory):
    config = config_factory()
    config['input_setting']['cif_path'] = '/some/path.cif'
    with pytest.raises(ValueError, match='Exactly one'):
        validate_config(config)


def test_neither_xml_nor_cif_path_raises(config_factory):
    config = config_factory()
    config['input_setting'] = {}
    with pytest.raises(ValueError, match='Exactly one'):
        validate_config(config)


@pytest.mark.parametrize('generation', [0, -1])
def test_non_positive_generation_raises(config_factory, generation):
    config = config_factory()
    config['structure']['generation'] = generation
    with pytest.raises(ValueError, match='generation'):
        validate_config(config)


def test_random_ratio_out_of_range_raises(config_factory):
    config = config_factory()
    config['structure']['random_ratio'] = 1.5
    with pytest.raises(ValueError, match='random_ratio'):
        validate_config(config)


def test_seed_absent_is_valid(config_factory):
    config = config_factory()
    config.pop('seed', None)
    validate_config(config)


@pytest.mark.parametrize('seed', [0, 1, 42])
def test_valid_seed_accepted(config_factory, seed):
    config = config_factory()
    config['seed'] = seed
    validate_config(config)


@pytest.mark.parametrize('seed', [-1, 1.5, '0', True])
def test_invalid_seed_raises(config_factory, seed):
    config = config_factory()
    config['seed'] = seed
    with pytest.raises(ValueError, match='seed'):
        validate_config(config)


def test_mutation_ratios_missing_key_raises(config_factory):
    config = config_factory()
    del config['structure']['mutation']['mutation_ratios']['split_site_ratio']
    with pytest.raises(ValueError, match='mutation_ratios'):
        validate_config(config)


def test_mutation_ratios_not_summing_to_one_raises(config_factory):
    config = config_factory()
    config['structure']['mutation']['mutation_ratios']['swap_same_mult_ratio'] = 0.9
    with pytest.raises(ValueError, match='sum to 1.0'):
        validate_config(config)


@pytest.mark.parametrize('optimizer_type', ['sequential', 'batch'])
def test_valid_optimizer_types_accepted(config_factory, optimizer_type):
    config = config_factory()
    config['optimizer']['type'] = optimizer_type
    validate_config(config)


def test_invalid_optimizer_type_raises(config_factory):
    config = config_factory()
    config['optimizer']['type'] = 'parallel'
    with pytest.raises(ValueError, match='optimizer.type'):
        validate_config(config)


def test_invalid_wavelength_raises(config_factory):
    config = config_factory()
    config['XRD_setting']['wave_length'] = 'NotAWavelength'
    with pytest.raises(ValueError, match='wave_length'):
        validate_config(config)


def test_two_theta_range_min_greater_than_max_raises(config_factory):
    config = config_factory()
    config['XRD_setting']['two_theta_range'] = [90.0, 1.0]
    with pytest.raises(ValueError, match='two_theta_range'):
        validate_config(config)


def test_invalid_normalize_raises(config_factory):
    config = config_factory()
    config['XRD_setting']['normalize'] = 'invalid'
    with pytest.raises(ValueError, match='normalize'):
        validate_config(config)


def test_scale_normalize_is_accepted(config_factory):
    config = config_factory()
    config['XRD_setting']['normalize'] = 'scale'
    validate_config(config)


def test_keep_max_non_positive_raises(config_factory):
    config = config_factory()
    config['structure']['dedup']['keep_max'] = 0
    with pytest.raises(ValueError, match='keep_max'):
        validate_config(config)


def test_dedup_wd_and_wasserstein_conflict_raises(config_factory):
    config = config_factory()
    config['structure']['dedup']['wd'] = 0.3
    with pytest.raises(ValueError, match='wasserstein'):
        validate_config(config)


def test_mix_ratios_not_summing_to_one_raises(config_factory):
    config = config_factory()
    config['structure']['dedup']['mix_ratios']['r_factor'] = 0.9
    with pytest.raises(ValueError, match='mix_ratios'):
        validate_config(config)


def test_add_no_symmetry_relax_non_boolean_raises(config_factory):
    config = config_factory()
    config['structure']['add_no_symmetry_relax'] = 'yes'
    with pytest.raises(ValueError, match='add_no_symmetry_relax'):
        validate_config(config)


def test_add_no_symmetry_relax_ratio_out_of_range_raises(config_factory):
    config = config_factory()
    config['structure']['add_no_symmetry_relax_ratio'] = 1.5
    with pytest.raises(ValueError, match='add_no_symmetry_relax_ratio'):
        validate_config(config)


def test_add_no_symmetry_relax_ratio_non_numeric_raises(config_factory):
    config = config_factory()
    config['structure']['add_no_symmetry_relax_ratio'] = 'half'
    with pytest.raises(ValueError, match='add_no_symmetry_relax_ratio'):
        validate_config(config)


def test_add_no_symmetry_relax_keys_are_optional(config_factory):
    config = config_factory()
    config['structure'].pop('add_no_symmetry_relax', None)
    config['structure'].pop('add_no_symmetry_relax_ratio', None)
    validate_config(config)


def test_validate_reference_info_accepts_complete_dict(sample_ref_info):
    validate_reference_info(sample_ref_info)


def test_validate_reference_info_missing_key_raises(sample_ref_info):
    del sample_ref_info['spg_num']
    with pytest.raises(ValueError, match='spg_num'):
        validate_reference_info(sample_ref_info)
