import pytest
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pyxtal.symmetry import Group
from pyxtal.tolerance import Tol_matrix

from xcape.modules import generation


# =============================================================================
# random_generation_pyxtal
# =============================================================================
def test_random_generation_pyxtal_returns_matching_length_lists(sample_ref_info, default_tol_matrix):
    structures, poscars, times = generation.random_generation_pyxtal(
        sample_ref_info, num_pop=2, gen=1, tol_matrix=default_tol_matrix,
    )
    assert len(structures) == len(poscars) == len(times) == 2


def test_random_generation_pyxtal_matches_requested_composition(sample_ref_info, default_tol_matrix):
    structures, _, _ = generation.random_generation_pyxtal(
        sample_ref_info, num_pop=2, gen=1, tol_matrix=default_tol_matrix,
    )
    for structure in structures:
        assert structure.composition.get_el_amt_dict() == {'Na': 4, 'Cl': 4}


def test_random_generation_pyxtal_poscar_has_gen_pop_comment(sample_ref_info, default_tol_matrix):
    _, poscars, _ = generation.random_generation_pyxtal(
        sample_ref_info, num_pop=2, gen=3, tol_matrix=default_tol_matrix,
    )
    for pop_idx, poscar in enumerate(poscars):
        assert poscar.splitlines()[0] == f'Gen: 3 / Pop: {pop_idx}'


# =============================================================================
# _generate_valid_random_structure
# =============================================================================
def test_generate_valid_random_structure_returns_pymatgen_structure(default_tol_matrix):
    from pymatgen.core import Structure

    structure = generation._generate_valid_random_structure(
        spg=Group(1),
        lattice=Lattice.from_para(8.0, 8.0, 8.0, 90.0, 90.0, 90.0, ltype='cubic'),
        species=['Na', 'Cl'],
        num_ions=[2, 2],
        tol_matrix=default_tol_matrix,
    )
    assert isinstance(structure, Structure)


def test_generate_valid_random_structure_raises_runtime_error_chained(monkeypatch, default_tol_matrix):
    def _always_fail(self, *args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(pyxtal, 'from_random', _always_fail)

    with pytest.raises(RuntimeError) as exc_info:
        generation._generate_valid_random_structure(
            spg=Group(1),
            lattice=Lattice.from_para(8.0, 8.0, 8.0, 90.0, 90.0, 90.0, ltype='cubic'),
            species=['Na', 'Cl'],
            num_ions=[2, 2],
            tol_matrix=default_tol_matrix,
            max_attempts=1,
        )
    assert 'after 1 attempts' in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)


# =============================================================================
# build_boltzmann_weights
# =============================================================================
def test_build_boltzmann_weights_uniform_when_all_metrics_inf(make_relax_result):
    candidates = [make_relax_result(energy=None) for _ in range(3)]
    weights = generation.build_boltzmann_weights(
        candidates, generation._create_metric_function('energy'), temperature=1.0,
    )
    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_build_boltzmann_weights_sums_to_one_and_favors_lower_metric(make_relax_result):
    candidates = [make_relax_result(energy=e) for e in (-2.0, 0.0, 2.0)]
    weights = generation.build_boltzmann_weights(
        candidates, generation._create_metric_function('energy'), temperature=1.0,
    )
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]


def test_build_boltzmann_weights_zero_temperature_selects_single_best(make_relax_result):
    candidates = [make_relax_result(energy=e) for e in (-2.0, 0.0, 2.0)]
    weights = generation.build_boltzmann_weights(
        candidates, generation._create_metric_function('energy'), temperature=0.0,
    )
    assert weights == [1.0, 0.0, 0.0]


def test_build_boltzmann_weights_zero_temperature_splits_ties(make_relax_result):
    candidates = [make_relax_result(energy=e) for e in (-1.0, -1.0, 2.0)]
    weights = generation.build_boltzmann_weights(
        candidates, generation._create_metric_function('energy'), temperature=0.0,
    )
    assert weights == pytest.approx([0.5, 0.5, 0.0])


# =============================================================================
# _create_metric_function
# =============================================================================
@pytest.mark.parametrize('metric_name,field', [
    ('energy', 'energy'),
    ('r_factor', 'r_factor'),
    ('r', 'r_factor'),
    ('wasserstein', 'wasserstein'),
    ('wd', 'wasserstein'),
    ('unknown_metric', 'energy'),
])
def test_create_metric_function_reads_expected_field(make_relax_result, metric_name, field):
    kwargs = {'energy': -1.0, 'r_factor': 0.1, 'wasserstein': 0.2}
    result = make_relax_result(**kwargs)
    metric_fn = generation._create_metric_function(metric_name)
    assert metric_fn(result) == pytest.approx(kwargs[field])


@pytest.mark.parametrize('metric_name,field', [
    ('energy', 'energy'),
    ('r_factor', 'r_factor'),
    ('wasserstein', 'wasserstein'),
])
def test_create_metric_function_returns_inf_when_field_none(make_relax_result, metric_name, field):
    result = make_relax_result(energy=None, r_factor=None, wasserstein=None)
    metric_fn = generation._create_metric_function(metric_name)
    assert metric_fn(result) == float('inf')


# =============================================================================
# _select_parent_index
# =============================================================================
def test_select_parent_index_always_weighted_when_ratio_is_one(make_relax_result):
    candidates = [make_relax_result(energy=0.0), make_relax_result(energy=0.0)]
    weights = [1.0, 0.0]
    seen = {generation._select_parent_index(candidates, weights, 1.0) for _ in range(50)}
    assert seen == {0}


def test_select_parent_index_falls_back_to_uniform_random_when_ratio_is_zero(make_relax_result):
    candidates = [make_relax_result(energy=0.0) for _ in range(5)]
    weights = [1.0, 0.0, 0.0, 0.0, 0.0]
    seen = {generation._select_parent_index(candidates, weights, 0.0) for _ in range(200)}
    assert len(seen) > 1


# =============================================================================
# _get_mutation_settings
# =============================================================================
def test_get_mutation_settings_reads_mutation_ratios(sample_config):
    settings = generation._get_mutation_settings(sample_config, gen=1)
    expected = sample_config['structure']['mutation']['mutation_ratios']
    assert settings['mutation_ratios'] == expected
    assert settings['max_attempts'] == 100
    assert settings['parent_metric'] == 'energy'
    assert settings['weighted_parent_ratio'] == pytest.approx(0.8)
    assert settings['init_spg_num'] is None


@pytest.mark.parametrize('gen,expected_temp', [
    (1, 1.0),
    (5, max(1.0 * 0.9 ** 4, 0.5)),
    (50, 0.5),
])
def test_get_mutation_settings_temperature_decay(sample_config, gen, expected_temp):
    settings = generation._get_mutation_settings(sample_config, gen=gen)
    assert settings['temperature'] == pytest.approx(expected_temp)


def test_get_mutation_settings_passes_through_spg_num(sample_config):
    settings = generation._get_mutation_settings(sample_config, gen=1, spg_num=225)
    assert settings['init_spg_num'] == 225


# =============================================================================
# _build_meta / _resolve_parent_id
# =============================================================================
def test_build_meta_uses_origin_type_for_mutate_name(make_relax_result):
    result = make_relax_result(energy=0.0, gen=2, pop=1)
    meta = generation._build_meta(result, 'swap_same_mult')
    assert meta == {
        'parent': 'none',
        'origin_type': 'swap_same_mult',
        'mutate_name': 'swap_same_mult',
        'orig_gen': 2,
        'orig_pop': 1,
    }


def test_resolve_parent_id_prefers_explicit_parent_field(make_relax_result):
    # RelaxResult.parent defaults to the string 'none' (not None), so this
    # branch wins even when gen/pop are set -- the gen_pop fallback below is
    # effectively unreachable unless parent is explicitly set to None.
    result = make_relax_result(energy=0.0, gen=3, pop=2)
    assert generation._resolve_parent_id(result) == 'none'


def test_resolve_parent_id_falls_back_to_gen_pop_when_parent_is_none(make_relax_result):
    result = make_relax_result(energy=0.0, gen=3, pop=2, parent=None)
    assert generation._resolve_parent_id(result) == '3_2'


def test_resolve_parent_id_returns_none_string_when_no_gen_pop(make_relax_result):
    result = make_relax_result(energy=0.0, parent=None)
    assert generation._resolve_parent_id(result) == 'none'


# =============================================================================
# append_previous_candidates
# =============================================================================
def test_append_previous_candidates_appends_and_tags_kept_best(make_relax_result, nacl_structure):
    candidates = [
        make_relax_result(structure=nacl_structure, energy=-1.0, gen=1, pop=0),
        make_relax_result(structure=nacl_structure, energy=-0.5, gen=1, pop=1),
    ]
    structure_list, poscar_list, t_gen_list = [], [], []
    meta = generation.append_previous_candidates(candidates, structure_list, poscar_list, t_gen_list, gen=2)

    assert len(structure_list) == len(poscar_list) == len(t_gen_list) == 2
    assert structure_list[0] is nacl_structure
    assert all(m['origin_type'] == 'kept_best' for m in meta)


# =============================================================================
# perform_mutation_and_append
# =============================================================================
def test_perform_mutation_and_append_num_mut_zero_is_noop(make_relax_result, nacl_structure, config_factory):
    candidates = [make_relax_result(structure=nacl_structure, energy=-1.0)]
    structure_list, poscar_list, t_gen_list = [], [], []
    meta = generation.perform_mutation_and_append(
        candidates, 0, structure_list, poscar_list, t_gen_list, gen=1,
        tol_matrix=Tol_matrix(prototype='atomic', factor=0.5), config=config_factory(),
    )
    assert meta == []
    assert structure_list == [] and poscar_list == [] and t_gen_list == []


def test_perform_mutation_and_append_empty_candidates_is_noop(config_factory):
    structure_list, poscar_list, t_gen_list = [], [], []
    meta = generation.perform_mutation_and_append(
        [], 2, structure_list, poscar_list, t_gen_list, gen=1,
        tol_matrix=Tol_matrix(prototype='atomic', factor=0.5), config=config_factory(),
    )
    assert meta == []
    assert structure_list == [] and poscar_list == [] and t_gen_list == []


def test_perform_mutation_and_append_appends_num_mut_entries_preserving_composition(
    movable_structure, make_relax_result, config_factory,
):
    candidates = [make_relax_result(structure=movable_structure, energy=-1.0, gen=1, pop=0)]
    structure_list, poscar_list, t_gen_list = [], [], []
    tol_matrix = Tol_matrix(prototype='atomic', factor=0.5)

    meta = generation.perform_mutation_and_append(
        candidates, num_mut=2, structure_list=structure_list, poscar_text_list=poscar_list,
        t_gen_list=t_gen_list, gen=1, tol_matrix=tol_matrix, config=config_factory(),
    )

    assert len(meta) == 2
    assert len(structure_list) == len(poscar_list) == len(t_gen_list) == 2
    for structure in structure_list:
        assert structure.composition.get_el_amt_dict() == {'Na': 2, 'Cl': 2}
    known_names = {
        'swap_same_mult', 'split', 'merge', 'swap_diff_elem',
        'reconfig', 'resample', 'multi_resample', 'max_attempts_reached',
    }
    assert all(m['mutate_name'] in known_names for m in meta)


# =============================================================================
# _prepare_structure_for_mutation -- spg-mismatch fallback (light smoke test)
# =============================================================================
def test_prepare_structure_for_mutation_falls_through_to_convert_when_resym_fails(
    monkeypatch, movable_structure,
):
    monkeypatch.setattr(generation, '_resymmetrize_to_target_group', lambda xtal, target: None)
    sentinel = object()
    monkeypatch.setattr(generation, '_convert_to_target_group', lambda xtal, target: sentinel)

    result = generation._prepare_structure_for_mutation(movable_structure, init_spg_num=225)
    assert result is sentinel
