import copy

import pytest
from pyxtal.tolerance import Tol_matrix

from xcape.modules import mutation
from xcape.modules.mutation import Mutator, sample_multiset, _all_multisets_sum_to

MUTATION_RATIOS = dict(
    swap_same_mult_ratio=0.25,
    split_site_ratio=0.1,
    merge_site_ratio=0.1,
    swap_diff_elem_ratio=0.15,
    site_reconfig_ratio=0.2,
    resample_pos_ratio=0.1,
    multi_resample_pos_ratio=0.1,
)

KNOWN_MUTATE_NAMES = {
    'swap_same_mult', 'split', 'merge', 'swap_diff_elem',
    'reconfig', 'resample', 'multi_resample',
}


@pytest.fixture
def mutator():
    return Mutator(**MUTATION_RATIOS, tolerance_matrix=Tol_matrix(prototype='atomic', factor=0.5))


# =============================================================================
# sample_multiset / _all_multisets_sum_to
# =============================================================================
def test_all_multisets_sum_to_enumerates_all_decompositions():
    assert _all_multisets_sum_to(12, (6, 12)) == [(6, 6), (12,)]


def test_sample_multiset_raises_when_no_decomposition_exists():
    with pytest.raises(ValueError):
        sample_multiset(7, (6, 12))


def test_sample_multiset_returns_a_valid_decomposition():
    result = sample_multiset(12, (6, 12))
    assert sum(result) == 12
    assert set(result) <= {6, 12}


def test_sample_multiset_weight_fn_biases_selection():
    def weight_fn(ms):
        return 1.0 if ms == (12,) else 0.0

    results = {sample_multiset(12, (6, 12), weight_fn=weight_fn) for _ in range(50)}
    assert results == {(12,)}


# =============================================================================
# Mutator.__init__ ratio validation
# =============================================================================
def test_mutator_raises_when_ratios_dont_sum_to_one():
    with pytest.raises(ValueError, match='sum of all move ratios'):
        Mutator(swap_same_mult_ratio=0.5, split_site_ratio=0.0, merge_site_ratio=0.0,
                 swap_diff_elem_ratio=0.0, site_reconfig_ratio=0.0,
                 resample_pos_ratio=0.0, multi_resample_pos_ratio=0.0)


def test_mutator_accepts_ratios_summing_to_one():
    m = Mutator(**MUTATION_RATIOS)
    assert m.max_attempts == 100


def test_mutator_default_ratios_sum_to_one():
    # Constructing a Mutator with no explicit ratios must not raise: the default
    # ratios sum to 1.0, so omitting `mutation_ratios` in the config is valid.
    m = Mutator()
    default_ratio_sum = (
        m.swap_same_mult_ratio
        + m.split_site_ratio
        + m.merge_site_ratio
        + m.swap_diff_elem_ratio
        + m.site_reconfig_ratio
        + m.resample_pos_ratio
        + m.multi_resample_pos_ratio
    )
    assert abs(default_ratio_sum - 1.0) <= 1e-6


# =============================================================================
# Moves that succeed on the P1 (movable) structure
# =============================================================================
def _composition_counts(structure):
    counts = {}
    for site in structure.atom_sites:
        counts[site.specie] = counts.get(site.specie, 0) + site.wp.multiplicity
    return counts


def test_swap_wyckoff_site_different_elements_swaps_species_when_changed(mutator, movable_pyxtal):
    for _ in range(20):
        structure = copy.deepcopy(movable_pyxtal)
        before_species = [s.specie for s in structure.atom_sites]
        before_counts = _composition_counts(structure)

        structure, is_changed = mutator.swap_wyckoff_site_different_elements(structure)

        assert isinstance(is_changed, bool)
        assert _composition_counts(structure) == before_counts
        if is_changed:
            after_species = [s.specie for s in structure.atom_sites]
            assert after_species != before_species
            return
    pytest.skip("swap_wyckoff_site_different_elements never succeeded in 20 attempts")


def test_reconfiguration_species_pick_is_sorted_for_reproducibility(
    mutator, movable_pyxtal, monkeypatch
):
    # The species to reconfigure is drawn with random.choice. Feeding it an
    # unsorted list({set}) makes the pick depend on PYTHONHASHSEED, so a seeded
    # run would not be reproducible across processes. The species must therefore
    # be sorted before the draw. We capture the first random.choice argument
    # (the species selection) and assert it is in sorted order.
    recorded = []
    real_choice = mutation.random.choice

    def recording_choice(seq):
        recorded.append(list(seq))
        return real_choice(seq)

    monkeypatch.setattr(mutation.random, 'choice', recording_choice)

    mutator.wyckoff_site_random_reconfiguration(copy.deepcopy(movable_pyxtal))

    assert recorded, "wyckoff_site_random_reconfiguration made no random.choice call"
    species_arg = recorded[0]
    assert set(species_arg) == {'Na', 'Cl'}
    assert species_arg == sorted(species_arg, key=str)


@pytest.mark.parametrize('method_name', [
    'wyckoff_site_random_reconfiguration',
    'resampling_atomic_positions',
    'multi_resampling_atomic_positions',
])
def test_position_mutations_succeed_and_preserve_composition(mutator, movable_pyxtal, method_name):
    method = getattr(mutator, method_name)
    for _ in range(20):
        structure = copy.deepcopy(movable_pyxtal)
        before_counts = _composition_counts(structure)
        before_positions = [tuple(s.position) for s in structure.atom_sites]

        structure, is_changed = method(structure)

        assert isinstance(is_changed, bool)
        assert _composition_counts(structure) == before_counts
        if is_changed:
            after_positions = [tuple(s.position) for s in structure.atom_sites]
            assert after_positions != before_positions
            return
    pytest.skip(f"{method_name} never succeeded in 20 attempts")


# =============================================================================
# Moves that are no-ops on the P1 structure (only one usable Wyckoff position)
# =============================================================================
@pytest.mark.parametrize('method_name', [
    'swap_wyckoff_site_same_multiplicity',
    'split_wyckoff_site',
    'merge_wyckoff_site',
])
def test_single_wyckoff_position_moves_are_graceful_noops(mutator, movable_pyxtal, method_name):
    method = getattr(mutator, method_name)
    structure, is_changed = method(copy.deepcopy(movable_pyxtal))
    assert is_changed is False


# =============================================================================
# Mutator.mutate
# =============================================================================
def test_mutate_always_returns_valid_three_tuple(mutator, movable_pyxtal):
    for _ in range(20):
        structure = copy.deepcopy(movable_pyxtal)
        result = mutator.mutate(structure)
        assert len(result) == 3
        mutated_structure, is_changed, mutate_name = result
        assert isinstance(is_changed, bool)
        assert mutate_name in KNOWN_MUTATE_NAMES


# =============================================================================
# _group_wps_by_multiplicity / _used_fixed_labels / _get_movable_site_indices
# =============================================================================
def test_group_wps_by_multiplicity_groups_p1_single_wp(mutator, movable_pyxtal):
    grouped = mutator._group_wps_by_multiplicity(movable_pyxtal.group)
    assert set(grouped.keys()) == {1}
    assert len(grouped[1]) == 1


def test_used_fixed_labels_empty_for_dof3_sites(mutator, movable_pyxtal):
    assert mutator._used_fixed_labels(movable_pyxtal) == []


def test_get_movable_site_indices_returns_all_sites_for_p1(mutator, movable_pyxtal):
    indices = mutator._get_movable_site_indices(movable_pyxtal)
    assert indices == list(range(len(movable_pyxtal.atom_sites)))
