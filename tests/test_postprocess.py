from unittest.mock import MagicMock

import pytest

from xcape.modules import postprocess
from xcape.utils.types import RelaxResult


def _result(energy=None, r_factor=None, wasserstein=None, **overrides):
    # copy_relax_result() calls structure.copy(), so give every result a
    # lightweight stand-in structure instead of None.
    kwargs = dict(
        structure=overrides.pop('structure', MagicMock()),
        energy=energy,
        status=overrides.pop('status', 'converged'),
        t_rel=0.0,
        r_factor=r_factor,
        wasserstein=wasserstein,
    )
    kwargs.update(overrides)
    return RelaxResult(**kwargs)


# =============================================================================
# _select_candidates
# =============================================================================
def test_select_candidates_keeps_results_within_any_enabled_delta(config_factory):
    config = config_factory()
    config['structure']['best_metric'] = {
        'energy': {'enabled': True, 'delta': 0.1},
        'r_factor': {'enabled': False, 'delta': 0.2},
        'wasserstein': {'enabled': False, 'delta': 0.5},
    }
    results = [
        _result(energy=-1.0, r_factor=0.5, wasserstein=1.0),   # best energy
        _result(energy=-0.95, r_factor=0.5, wasserstein=1.0),  # within delta
        _result(energy=0.0, r_factor=0.01, wasserstein=0.01),  # outside energy delta, r/wd disabled
    ]
    selected = postprocess._select_candidates(results, config)
    assert len(selected) == 2
    assert {r.energy for r in selected} == {-1.0, -0.95}


def test_select_candidates_legacy_scalar_threshold_keys(config_factory):
    config = config_factory()
    config['structure']['best_metric'] = {'Energy': 0.05}
    results = [_result(energy=-1.0), _result(energy=-0.9)]
    selected = postprocess._select_candidates(results, config)
    assert len(selected) == 1
    assert selected[0].energy == -1.0


def test_select_candidates_returns_copies_not_originals(config_factory):
    config = config_factory()
    results = [_result(energy=-1.0)]
    selected = postprocess._select_candidates(results, config)
    assert selected[0] is not results[0]
    assert selected[0].energy == results[0].energy


# =============================================================================
# deduplicate_candidates / _check_duplicate
# =============================================================================
def test_deduplicate_candidates_removes_metric_and_structure_duplicate():
    structure_matcher = MagicMock()
    structure_matcher.fit.return_value = True

    candidates = [
        _result(energy=-1.0, r_factor=0.01, wasserstein=0.01, gen=1, pop=0),
        _result(energy=-1.0005, r_factor=0.011, wasserstein=0.012, gen=1, pop=1),
    ]
    dedup_cfg = {'energy': 0.005, 'r_factor': 0.05, 'wasserstein': 0.5}

    unique = postprocess.deduplicate_candidates(candidates, dedup_cfg, structure_matcher)
    assert len(unique) == 1


def test_deduplicate_candidates_keeps_structurally_different_candidates():
    structure_matcher = MagicMock()
    structure_matcher.fit.return_value = False

    candidates = [
        _result(energy=-1.0, r_factor=0.01, wasserstein=0.01, gen=1, pop=0),
        _result(energy=-1.0005, r_factor=0.011, wasserstein=0.012, gen=1, pop=1),
    ]
    dedup_cfg = {'energy': 0.005, 'r_factor': 0.05, 'wasserstein': 0.5}

    unique = postprocess.deduplicate_candidates(candidates, dedup_cfg, structure_matcher)
    assert len(unique) == 2


def test_deduplicate_candidates_skips_results_with_missing_metrics():
    structure_matcher = MagicMock()
    structure_matcher.fit.return_value = True
    candidates = [
        _result(energy=None, r_factor=0.01, wasserstein=0.01, gen=1, pop=0),
        _result(energy=-1.0, r_factor=0.01, wasserstein=0.01, gen=1, pop=1),
    ]
    unique = postprocess.deduplicate_candidates(candidates, {}, structure_matcher)
    assert len(unique) == 2


def test_check_duplicate_swallows_structure_matcher_exceptions():
    structure_matcher = MagicMock()
    structure_matcher.fit.side_effect = RuntimeError("boom")
    candidate = _result(energy=-1.0, r_factor=0.01, wasserstein=0.01)
    existing = _result(energy=-1.0, r_factor=0.01, wasserstein=0.01, gen=1, pop=0)
    is_dup = postprocess._check_duplicate(candidate, [existing], 0.005, 0.05, 0.5, structure_matcher)
    assert is_dup is False


# =============================================================================
# _apply_keep_max_mix
# =============================================================================
def test_apply_keep_max_mix_returns_unchanged_when_under_limit():
    candidates = [_result(energy=float(i)) for i in range(3)]
    result = postprocess._apply_keep_max_mix(candidates, keep_max=5, mix_ratios=None)
    assert result == candidates


def test_apply_keep_max_mix_returns_unchanged_when_keep_max_none():
    candidates = [_result(energy=float(i)) for i in range(3)]
    assert postprocess._apply_keep_max_mix(candidates, keep_max=None) is candidates


def test_apply_keep_max_mix_truncates_to_keep_max():
    candidates = [
        _result(energy=float(i), r_factor=float(i), wasserstein=float(i))
        for i in range(10)
    ]
    result = postprocess._apply_keep_max_mix(
        candidates, keep_max=5,
        mix_ratios={'r_factor': 0.4, 'energy': 0.4, 'wasserstein': 0.2},
    )
    assert len(result) == 5


# =============================================================================
# write_outputs
# =============================================================================
def test_write_outputs_appends_poscar_and_contcar(tmp_path):
    postprocess.write_outputs(tmp_path, ['poscar-1\n'], ['contcar-1\n'])
    postprocess.write_outputs(tmp_path, ['poscar-2\n'], ['contcar-2\n'])

    assert (tmp_path / 'POSCARs').read_text() == 'poscar-1\nposcar-2\n'
    assert (tmp_path / 'CONTCARs').read_text() == 'contcar-1\ncontcar-2\n'


def test_write_no_symmetry_outputs_uses_a_separate_file(tmp_path):
    postprocess.write_outputs(tmp_path, ['poscar-1\n'], ['contcar-1\n'])
    postprocess.write_no_symmetry_outputs(tmp_path, ['nosym-1\n'])
    postprocess.write_no_symmetry_outputs(tmp_path, ['nosym-2\n'])

    assert (tmp_path / 'CONTCARs_no_sym').read_text() == 'nosym-1\nnosym-2\n'
    # The symmetry-constrained CONTCARs stay untouched.
    assert (tmp_path / 'CONTCARs').read_text() == 'contcar-1\n'


# =============================================================================
# process_no_symmetry_results
# =============================================================================
def _no_sym_args(tmp_path, results, monkeypatch, config):
    monkeypatch.setattr(
        postprocess, 'poscar_text_from_pymatgen',
        lambda structure, gen, pop: f'CONTCAR {gen}_{pop}\n',
    )
    monkeypatch.setattr(
        postprocess, 'analyze_relaxed_structure',
        lambda *args, **kwargs: (0.25, 1.5),
    )
    return dict(
        no_symmetry_relax_results=results,
        gen=3,
        output_path=tmp_path,
        no_symmetry_log_file=tmp_path / 'non-sym.log',
        xrd_calc=MagicMock(),
        two_theta_ref=[10.0], intensity_ref=[100.0],
        tt_tol=0.3, two_theta_range=(1.0, 90.0),
        config=config,
    )


def test_process_no_symmetry_results_writes_contcars_and_log(tmp_path, monkeypatch, config_factory):
    results = [_result(energy=-2.0, pop=1), _result(energy=-1.0, pop=4)]
    postprocess.process_no_symmetry_results(
        **_no_sym_args(tmp_path, results, monkeypatch, config_factory())
    )

    # CONTCARs are keyed by the generation and the source population index.
    assert (tmp_path / 'CONTCARs_no_sym').read_text() == 'CONTCAR 3_1\nCONTCAR 3_4\n'

    log_rows = (tmp_path / 'non-sym.log').read_text().splitlines()
    assert len(log_rows) == 2
    assert log_rows[0].split()[:2] == ['3', '1']
    assert log_rows[1].split()[:2] == ['3', '4']
    # Metrics are computed for these results too.
    assert [r.r_factor for r in results] == [0.25, 0.25]
    assert [r.wasserstein for r in results] == [1.5, 1.5]


def test_process_no_symmetry_results_penalises_failed_relaxations(
    tmp_path, monkeypatch, config_factory
):
    results = [_result(energy=-2.0, pop=0, status='distance_violation_final')]
    postprocess.process_no_symmetry_results(
        **_no_sym_args(tmp_path, results, monkeypatch, config_factory())
    )

    assert results[0].r_factor == postprocess.FAILED_METRIC_PENALTY
    assert results[0].wasserstein == postprocess.FAILED_METRIC_PENALTY
    assert results[0].energy == postprocess.FAILED_METRIC_PENALTY


def test_process_no_symmetry_results_empty_writes_nothing(tmp_path, monkeypatch, config_factory):
    postprocess.process_no_symmetry_results(
        **_no_sym_args(tmp_path, [], monkeypatch, config_factory())
    )
    assert not (tmp_path / 'CONTCARs_no_sym').exists()
    assert not (tmp_path / 'non-sym.log').exists()


# =============================================================================
# postprocess_generation wiring for the non-symmetry gate
# =============================================================================
def _postprocess_generation_args(tmp_path, monkeypatch, config, **overrides):
    monkeypatch.setattr(
        postprocess, 'poscar_text_from_pymatgen',
        lambda structure, gen, pop: f'CONTCAR {gen}_{pop}\n',
    )
    monkeypatch.setattr(
        postprocess, 'analyze_relaxed_structure',
        lambda *args, **kwargs: (0.25, 1.5),
    )
    matcher = MagicMock()
    matcher.fit.return_value = False
    kwargs = dict(
        relax_results=[],
        no_symmetry_relax_results=[],
        poscar_text_list=[],
        appended_meta=[],
        previous_candidates=[],
        gen=2,
        env={
            'output_path': tmp_path,
            'xrd_calc': MagicMock(),
            'structure_matcher': matcher,
        },
        config=config,
        two_theta_ref=[10.0], intensity_ref=[100.0],
        tt_tol=0.3, two_theta_range=(1.0, 90.0),
        log_file=tmp_path / 'generation.log',
        no_symmetry_log_file=tmp_path / 'non-sym.log',
    )
    kwargs.update(overrides)
    return kwargs


def test_postprocess_generation_keeps_no_symmetry_results_out_of_the_pool(
    tmp_path, monkeypatch, config_factory
):
    config = config_factory()
    # The unconstrained result has by far the best energy, so if it leaked into
    # the selection pool it would dominate the candidates returned here.
    sym_results = [_result(energy=-1.0, pop=0)]
    no_sym_results = [_result(energy=-50.0, pop=0)]

    candidates = postprocess.postprocess_generation(
        **_postprocess_generation_args(
            tmp_path, monkeypatch, config,
            relax_results=sym_results,
            no_symmetry_relax_results=no_sym_results,
        )
    )

    assert [c.energy for c in candidates] == [-1.0]
    # ...but it is still recorded in its own files.
    assert (tmp_path / 'CONTCARs_no_sym').read_text() == 'CONTCAR 2_0\n'
    assert len((tmp_path / 'non-sym.log').read_text().splitlines()) == 1
    # ...and does not pollute the main generation log.
    assert len((tmp_path / 'generation.log').read_text().splitlines()) == 1


def test_postprocess_generation_requires_log_path_when_results_present(
    tmp_path, monkeypatch, config_factory
):
    with pytest.raises(ValueError, match='no_symmetry_log_file is required'):
        postprocess.postprocess_generation(
            **_postprocess_generation_args(
                tmp_path, monkeypatch, config_factory(),
                relax_results=[_result(energy=-1.0, pop=0)],
                no_symmetry_relax_results=[_result(energy=-2.0, pop=0)],
                no_symmetry_log_file=None,
            )
        )


def test_postprocess_generation_skips_no_symmetry_files_when_gate_is_off(
    tmp_path, monkeypatch, config_factory
):
    postprocess.postprocess_generation(
        **_postprocess_generation_args(
            tmp_path, monkeypatch, config_factory(),
            relax_results=[_result(energy=-1.0, pop=0)],
        )
    )
    assert not (tmp_path / 'CONTCARs_no_sym').exists()
    assert not (tmp_path / 'non-sym.log').exists()


# =============================================================================
# _apply_appended_metadata / _mark_kept_candidates
# =============================================================================
def test_apply_appended_metadata_attaches_tail_entries():
    results = [_result(energy=0.0), _result(energy=-1.0), _result(energy=-2.0)]
    meta = [
        {'parent': '1_0', 'origin_type': 'swap_same_mult'},
        {'parent': '1_1', 'origin_type': 'split_site'},
    ]
    postprocess._apply_appended_metadata(results, meta)

    assert results[0].parent == 'none'  # untouched, no metadata for this one
    assert results[1].parent == '1_0'
    assert results[1].origin_type == 'swap_same_mult'
    assert results[2].parent == '1_1'
    assert results[2].origin_type == 'split_site'


def test_apply_appended_metadata_noop_for_empty_meta():
    results = [_result(energy=0.0, parent='none')]
    postprocess._apply_appended_metadata(results, [])
    assert results[0].parent == 'none'


def test_mark_kept_candidates_resets_timing_and_lineage():
    results = [_result(energy=0.0, gen=3, pop=2, t_gen=5.0, t_rel=5.0)]
    postprocess._mark_kept_candidates(results)
    assert results[0].origin_type == 'kept_best'
    assert results[0].parent == '3_2'
    assert results[0].t_gen == 0.0
    assert results[0].t_rel == 0.0


# =============================================================================
# analyze_relaxed_structure
# =============================================================================
def test_analyze_relaxed_structure_returns_zero_r_factor_for_identical_pattern():
    xrd_calc = MagicMock()
    pattern = MagicMock()
    pattern.x = [10.0, 20.0]
    pattern.y = [100.0, 50.0]
    xrd_calc.get_pattern.return_value = pattern

    r_factor, wasserstein = postprocess.analyze_relaxed_structure(
        structure=MagicMock(),
        two_theta_ref=[10.0, 20.0],
        intensity_ref=[100.0, 50.0],
        xrd_calc=xrd_calc,
        tt_tol=0.1,
        two_theta_range=(0.0, 30.0),
    )
    assert r_factor == pytest.approx(0.0)
    assert wasserstein == pytest.approx(0.0, abs=1e-6)
