from unittest.mock import MagicMock

import pytest

from xcape import main
from xcape.utils.types import RelaxResult
from xcape.validation import DEFAULT_EARLY_STOPPING_THRESHOLD


# =============================================================================
# early_stopping
# =============================================================================
def test_early_stopping_false_when_no_candidates_below_threshold(make_relax_result):
    candidates = [make_relax_result(r_factor=0.5), make_relax_result(r_factor=0.3)]
    assert main.early_stopping(candidates, threshold=0.1) is False


def test_early_stopping_true_when_one_candidate_below_threshold(make_relax_result):
    candidates = [make_relax_result(r_factor=0.5), make_relax_result(r_factor=0.05)]
    assert main.early_stopping(candidates, threshold=0.1) is True


def test_early_stopping_true_when_multiple_candidates_below_threshold(make_relax_result):
    candidates = [make_relax_result(r_factor=0.05), make_relax_result(r_factor=0.02)]
    assert main.early_stopping(candidates, threshold=0.1) is True


def test_early_stopping_ignores_none_r_factor(make_relax_result):
    candidates = [make_relax_result(r_factor=None)]
    assert main.early_stopping(candidates, threshold=100.0) is False


# =============================================================================
# rss_mb
# =============================================================================
def test_rss_mb_returns_positive_float():
    value = main.rss_mb()
    assert isinstance(value, float)
    assert value > 0


# =============================================================================
# _get_early_stopping_threshold
# =============================================================================
def test_get_early_stopping_threshold_default_when_absent(config_factory):
    config = config_factory()
    del config['structure']['early_stopping_threshold']
    assert main._get_early_stopping_threshold(config) == DEFAULT_EARLY_STOPPING_THRESHOLD


def test_get_early_stopping_threshold_uses_explicit_override(config_factory):
    config = config_factory()
    config['structure']['early_stopping_threshold'] = 0.01
    assert main._get_early_stopping_threshold(config) == pytest.approx(0.01)


# =============================================================================
# _resolve_two_theta_range
# =============================================================================
def test_resolve_two_theta_range_uses_explicit_config_range():
    xrd_cfg = {'two_theta_range': [10.0, 80.0], 'two_theta_tol': 0.3}
    result = main._resolve_two_theta_range([5.0, 90.0], xrd_cfg)
    assert result == (9.7, 80.3)


def test_resolve_two_theta_range_falls_back_to_reference_extent():
    xrd_cfg = {'two_theta_tol': 0.3}
    result = main._resolve_two_theta_range([15.0, 60.0, 40.0], xrd_cfg)
    assert result == pytest.approx((14.7, 60.3))


# =============================================================================
# _build_distance_constraint_tol_matrix
# =============================================================================
def test_build_distance_constraint_tol_matrix_sequential_uses_env_mlip(config_factory, monkeypatch):
    config = config_factory()
    config['optimizer']['type'] = 'sequential'
    env = {'mlip': 'env-mlip-sentinel'}
    ref_info = {}
    captured = {}

    def fake_ensure(cfg, calc):
        captured['calc'] = calc
        return 'tol-matrix-sentinel'

    monkeypatch.setattr(main, 'ensure_distance_constraint', fake_ensure)

    result = main._build_distance_constraint_tol_matrix(config, env)

    assert captured['calc'] == 'env-mlip-sentinel'
    assert result == 'tol-matrix-sentinel'


def test_build_distance_constraint_tol_matrix_batch_uses_temporary_mlip(config_factory, monkeypatch):
    config = config_factory()
    config['optimizer']['type'] = 'batch'
    env = {'mlip': 'env-mlip-sentinel'}
    ref_info = {}
    temp_mlip = object()
    captured = {}

    def fake_load(cfg, optimizer_type=None):
        captured['optimizer_type'] = optimizer_type
        return temp_mlip

    def fake_ensure(cfg, calc):
        captured['calc'] = calc
        return 'tol-matrix-sentinel'

    monkeypatch.setattr(main, 'load_sevennet_calc', fake_load)
    monkeypatch.setattr(main, 'ensure_distance_constraint', fake_ensure)

    result = main._build_distance_constraint_tol_matrix(config, env)

    assert captured['optimizer_type'] == 'sequential'
    assert captured['calc'] is temp_mlip
    assert captured['calc'] != 'env-mlip-sentinel'
    assert result == 'tol-matrix-sentinel'


# =============================================================================
# _build_generation_population
# =============================================================================
def _fake_generator(ref_info, num_rand_pop, gen, tol_matrix):
    return (
        [f'struct{i}' for i in range(num_rand_pop)],
        [f'poscar{i}' for i in range(num_rand_pop)],
        [0.0] * num_rand_pop,
    )


def test_build_generation_population_no_candidates_uses_all_random(config_factory, monkeypatch, make_relax_result):
    config = config_factory()
    config['structure']['population'] = 8
    env = {'generator': _fake_generator}

    def fail_if_called(**kwargs):
        raise AssertionError('perform_mutation_and_append should not be called')

    monkeypatch.setattr(main, 'perform_mutation_and_append', fail_if_called)

    structure_list, poscar_text_list, t_gen_list, appended_meta = main._build_generation_population(
        env=env, ref_info={'spg_num': 225}, candidates=[], gen=1, tol_matrix=None, config=config,
    )

    assert len(structure_list) == 8
    assert len(poscar_text_list) == 8
    assert len(t_gen_list) == 8
    assert appended_meta == []


def test_build_generation_population_with_candidates_splits_and_mutates(config_factory, monkeypatch, make_relax_result):
    config = config_factory()
    config['structure']['population'] = 10
    config['structure']['random_ratio'] = 0.2
    env = {'generator': _fake_generator}
    candidates = [make_relax_result(r_factor=0.1)]

    captured = {}

    def fake_mutate(**kwargs):
        captured.update(kwargs)
        return [{'parent': '1_0', 'origin_type': 'swap_same_mult'}]

    monkeypatch.setattr(main, 'perform_mutation_and_append', fake_mutate)

    structure_list, poscar_text_list, t_gen_list, appended_meta = main._build_generation_population(
        env=env, ref_info={'spg_num': 225}, candidates=candidates, gen=2, tol_matrix='tm', config=config,
    )

    assert len(structure_list) == 2  # num_rand_pop = int(10 * 0.2)
    assert captured['num_mut'] == 8  # 10 - 2
    assert captured['candidates'] is candidates
    assert captured['spg_num'] == 225
    assert appended_meta == [{'parent': '1_0', 'origin_type': 'swap_same_mult'}]


# =============================================================================
# _run_relaxation
# =============================================================================
def test_run_relaxation_sequential_calls_serial_relax(config_factory, monkeypatch):
    config = config_factory()
    config['optimizer']['type'] = 'sequential'
    monkeypatch.setattr(main, 'serial_relax', lambda *args, **kwargs: 'serial-sentinel')
    monkeypatch.setattr(main, 'batch_relax', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('should not be called')))

    result = main._run_relaxation([], [], {'mlip': None}, None, config)
    assert result == 'serial-sentinel'


def test_run_relaxation_batch_calls_batch_relax(config_factory, monkeypatch):
    config = config_factory()
    config['optimizer']['type'] = 'batch'
    monkeypatch.setattr(main, 'batch_relax', lambda *args, **kwargs: 'batch-sentinel')
    monkeypatch.setattr(main, 'serial_relax', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('should not be called')))

    result = main._run_relaxation([], [], {'mlip': None}, None, config)
    assert result == 'batch-sentinel'


def test_run_relaxation_unsupported_type_raises(config_factory):
    config = config_factory()
    config['optimizer']['type'] = 'parallel'
    with pytest.raises(ValueError, match='Unsupported optimizer type'):
        main._run_relaxation([], [], {'mlip': None}, None, config)


# =============================================================================
# _run_generation (build -> relax -> postprocess wiring)
# =============================================================================
def test_run_generation_wires_build_relax_postprocess(monkeypatch):
    monkeypatch.setattr(
        main, '_build_generation_population',
        lambda **kwargs: (['struct'], ['poscar'], [0.1], [{'meta': 1}]),
    )
    monkeypatch.setattr(
        main, '_run_relaxation',
        lambda structure_list, t_gen_list, env, tol_matrix, config: ['relaxed'],
    )
    captured = {}

    def fake_postprocess(**kwargs):
        captured.update(kwargs)
        return ['candidate']

    monkeypatch.setattr(main, '_postprocess_generation_step', fake_postprocess)

    candidates, no_sym_results = main._run_generation(
        gen=1, env={'x': 1}, ref_info={}, candidates=[], tol_matrix='tm',
        config={}, two_theta_ref=[1.0], intensity_ref=[2.0], tt_tol=0.3,
        two_theta_range=(1.0, 90.0), log_file='log',
        no_symmetry_log_file='non-sym-log',
    )

    assert candidates == ['candidate']
    # The gate is off by default, so no second pool comes back.
    assert no_sym_results == []
    # The relaxation output and generation artifacts flow into postprocess.
    assert captured['relax_results'] == ['relaxed']
    assert captured['poscar_text_list'] == ['poscar']
    assert captured['appended_meta'] == [{'meta': 1}]
    assert captured['gen'] == 1
    # The gate is off by default, so nothing extra is relaxed.
    assert captured['no_symmetry_relax_results'] == []


# =============================================================================
# Non-symmetry relaxation gate
# =============================================================================
def _relax_result(energy, status='converged'):
    return RelaxResult(
        structure=MagicMock(), energy=energy, status=status, t_rel=0.0, t_gen=0.0,
    )


def test_select_no_symmetry_targets_picks_lowest_energy_converged(config_factory):
    config = config_factory()
    config['structure']['add_no_symmetry_relax_ratio'] = 0.5
    results = [
        _relax_result(-1.0),
        _relax_result(-3.0),
        _relax_result(-2.0),
        _relax_result(-0.5),
    ]
    assert main._select_no_symmetry_targets(results, config) == [1, 2]


def test_select_no_symmetry_targets_excludes_failed_relaxations(config_factory):
    config = config_factory()
    config['structure']['add_no_symmetry_relax_ratio'] = 0.5
    results = [
        # A diverged trial can carry an unphysically low energy at this point,
        # because the failure penalty is only applied during post-processing.
        _relax_result(-99.0, status='distance_violation_final'),
        _relax_result(-3.0),
        _relax_result(-2.0),
        _relax_result(None, status='not_converged'),
    ]
    assert main._select_no_symmetry_targets(results, config) == [1, 2]


def test_select_no_symmetry_targets_keeps_at_least_one(config_factory):
    config = config_factory()
    config['structure']['add_no_symmetry_relax_ratio'] = 0.1
    # int(2 * 0.1) == 0, but the gate must not silently do nothing.
    results = [_relax_result(-1.0), _relax_result(-2.0)]
    assert main._select_no_symmetry_targets(results, config) == [1]


def test_select_no_symmetry_targets_empty_for_zero_ratio(config_factory):
    config = config_factory()
    config['structure']['add_no_symmetry_relax_ratio'] = 0.0
    assert main._select_no_symmetry_targets([_relax_result(-1.0)], config) == []


def test_select_no_symmetry_targets_defaults_when_ratio_absent(config_factory):
    config = config_factory()
    config['structure'].pop('add_no_symmetry_relax_ratio', None)
    results = [_relax_result(float(-i)) for i in range(10)]
    # Falls back to DEFAULT_ADD_NO_SYMMETRY_RELAX_RATIO (0.3) instead of raising.
    assert len(main._select_no_symmetry_targets(results, config)) == 3


def test_run_no_symmetry_relaxation_disabled_returns_empty(config_factory):
    config = config_factory()
    config['structure']['add_no_symmetry_relax'] = False
    result = main._run_no_symmetry_relaxation(
        relax_results=[_relax_result(-1.0)], structure_list=['s'], t_gen_list=[0.1],
        gen=1, env={}, tol_matrix=None, config=config,
    )
    assert result == []


def test_run_no_symmetry_relaxation_does_not_mutate_shared_config(config_factory, monkeypatch):
    """A shallow config.copy() would disable fix_symmetry for every later generation."""
    config = config_factory()
    config['structure']['add_no_symmetry_relax'] = True
    config['structure']['add_no_symmetry_relax_ratio'] = 1.0
    config['optimizer']['fix_symmetry'] = True

    seen_fix_symmetry = []

    def fake_relax(structure_list, t_gen_list, env, tol_matrix, cfg):
        seen_fix_symmetry.append(cfg['optimizer']['fix_symmetry'])
        return [_relax_result(-1.0) for _ in structure_list]

    monkeypatch.setattr(main, '_run_relaxation', fake_relax)

    main._run_no_symmetry_relaxation(
        relax_results=[_relax_result(-1.0)], structure_list=['s'], t_gen_list=[0.1],
        gen=1, env={}, tol_matrix=None, config=config,
    )

    assert seen_fix_symmetry == [False]
    assert config['optimizer']['fix_symmetry'] is True


def test_run_no_symmetry_relaxation_relaxes_generated_structures_and_tags_source(
    config_factory, monkeypatch
):
    config = config_factory()
    config['structure']['add_no_symmetry_relax'] = True
    config['structure']['add_no_symmetry_relax_ratio'] = 0.5

    captured = {}

    def fake_relax(structure_list, t_gen_list, env, tol_matrix, cfg):
        captured['structures'] = structure_list
        captured['t_gen'] = t_gen_list
        return [_relax_result(-9.0) for _ in structure_list]

    monkeypatch.setattr(main, '_run_relaxation', fake_relax)

    relax_results = [
        _relax_result(-1.0), _relax_result(-3.0), _relax_result(-2.0), _relax_result(-0.5),
    ]
    results = main._run_no_symmetry_relaxation(
        relax_results=relax_results,
        structure_list=['s0', 's1', 's2', 's3'],
        t_gen_list=[0.0, 0.1, 0.2, 0.3],
        gen=7, env={}, tol_matrix=None, config=config,
    )

    # The originally generated structures are re-relaxed, not the relaxed output.
    assert captured['structures'] == ['s1', 's2']
    assert captured['t_gen'] == [0.1, 0.2]
    # Each result stays addressable back to its symmetry-constrained counterpart.
    assert [r.gen for r in results] == [7, 7]
    assert [r.pop for r in results] == [1, 2]
    assert [r.parent for r in results] == ['7_1', '7_2']


def test_run_no_symmetry_relaxation_no_converged_candidate_returns_empty(
    config_factory, monkeypatch
):
    config = config_factory()
    config['structure']['add_no_symmetry_relax'] = True
    monkeypatch.setattr(
        main, '_run_relaxation',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('should not relax')),
    )
    result = main._run_no_symmetry_relaxation(
        relax_results=[_relax_result(-1.0, status='not_converged')],
        structure_list=['s'], t_gen_list=[0.1],
        gen=1, env={}, tol_matrix=None, config=config,
    )
    assert result == []


# =============================================================================
# main.main (top-level loop: generations, early stopping, memory log)
# =============================================================================
def _patch_main_setup(monkeypatch, config, out, ref_info):
    """Stub every heavy setup step so main() runs pure orchestration."""
    out.mkdir(parents=True, exist_ok=True)
    env = {
        'output_path': out, 'generator': None, 'mlip': None,
        'xrd_calc': None, 'structure_matcher': None,
    }
    monkeypatch.setattr(main, 'load_config', lambda path: config)
    monkeypatch.setattr(main, 'setup_logging', lambda output_path: None)
    monkeypatch.setattr(main, 'set_global_seeds', lambda seed: None)
    monkeypatch.setattr(main, 'setup_environment', lambda cfg: env)
    monkeypatch.setattr(main, 'prepare_reference', lambda cfg: ref_info)
    monkeypatch.setattr(main, '_build_distance_constraint_tol_matrix', lambda cfg, e: 'tol')
    monkeypatch.setattr(main, 'dump_yaml', lambda path, cfg: None)
    monkeypatch.setattr(main, 'initialize_generation_log', lambda path: None)


def test_main_runs_all_generations_when_no_early_stop(
    config_factory, monkeypatch, tmp_path, make_relax_result,
):
    config = config_factory()
    config['structure']['generation'] = 3
    config['structure']['early_stopping_threshold'] = 0.0  # unreachable
    out = tmp_path / 'run'
    config['output_path'] = str(out)
    ref_info = {'two_theta': [10.0, 20.0], 'intensity': [1.0, 2.0], 'spg_num': 225}
    _patch_main_setup(monkeypatch, config, out, ref_info)

    gens_run = []

    def fake_run_generation(**kwargs):
        gens_run.append(kwargs['gen'])
        return [make_relax_result(r_factor=0.5)], []  # never below threshold

    monkeypatch.setattr(main, '_run_generation', fake_run_generation)

    assert main.main(config_path='ignored') == 0
    assert gens_run == [1, 2, 3]
    # memory.log gets one line appended per generation.
    mem_lines = (out / 'memory.log').read_text().strip().splitlines()
    assert len(mem_lines) == 3


def test_main_stops_early_when_threshold_met(
    config_factory, monkeypatch, tmp_path, make_relax_result,
):
    config = config_factory()
    config['structure']['generation'] = 5
    config['structure']['early_stopping_threshold'] = 0.01
    out = tmp_path / 'run'
    config['output_path'] = str(out)
    ref_info = {'two_theta': [10.0, 20.0], 'intensity': [1.0, 2.0], 'spg_num': 225}
    _patch_main_setup(monkeypatch, config, out, ref_info)

    gens_run = []

    def fake_run_generation(**kwargs):
        gen = kwargs['gen']
        gens_run.append(gen)
        r_factor = 0.005 if gen == 2 else 0.5  # crosses threshold at gen 2
        return [make_relax_result(r_factor=r_factor)], []

    monkeypatch.setattr(main, '_run_generation', fake_run_generation)

    assert main.main(config_path='ignored') == 0
    assert gens_run == [1, 2]  # loop halts after the early-stop generation


def test_main_stops_early_on_a_non_symmetry_match(
    config_factory, monkeypatch, tmp_path, make_relax_result,
):
    """A gate result below the threshold is a solved pattern, so the run stops."""
    config = config_factory()
    config['structure']['generation'] = 5
    config['structure']['early_stopping_threshold'] = 0.01
    out = tmp_path / 'run'
    config['output_path'] = str(out)
    ref_info = {'two_theta': [10.0, 20.0], 'intensity': [1.0, 2.0], 'spg_num': 225}
    _patch_main_setup(monkeypatch, config, out, ref_info)

    gens_run = []

    def fake_run_generation(**kwargs):
        gen = kwargs['gen']
        gens_run.append(gen)
        # The candidate pool never converges; only the unconstrained pool does.
        no_sym = [make_relax_result(r_factor=0.005)] if gen == 3 else []
        return [make_relax_result(r_factor=0.5)], no_sym

    monkeypatch.setattr(main, '_run_generation', fake_run_generation)

    assert main.main(config_path='ignored') == 0
    assert gens_run == [1, 2, 3]


def test_main_does_not_stop_on_penalised_no_symmetry_results(
    config_factory, monkeypatch, tmp_path, make_relax_result,
):
    """Failed gate relaxations carry the 99.9 penalty and must not trigger a stop."""
    config = config_factory()
    config['structure']['generation'] = 2
    config['structure']['early_stopping_threshold'] = 0.01
    out = tmp_path / 'run'
    config['output_path'] = str(out)
    ref_info = {'two_theta': [10.0, 20.0], 'intensity': [1.0, 2.0], 'spg_num': 225}
    _patch_main_setup(monkeypatch, config, out, ref_info)

    gens_run = []

    def fake_run_generation(**kwargs):
        gens_run.append(kwargs['gen'])
        return (
            [make_relax_result(r_factor=0.5)],
            [make_relax_result(r_factor=99.9, status='not_converged')],
        )

    monkeypatch.setattr(main, '_run_generation', fake_run_generation)

    assert main.main(config_path='ignored') == 0
    assert gens_run == [1, 2]


# =============================================================================
# _early_stopping_reason
# =============================================================================
def test_early_stopping_reason_none_when_neither_pool_converges(make_relax_result):
    assert main._early_stopping_reason(
        [make_relax_result(r_factor=0.5)], [make_relax_result(r_factor=0.4)], 0.01
    ) is None


def test_early_stopping_reason_names_the_candidate_pool(make_relax_result):
    reason = main._early_stopping_reason(
        [make_relax_result(r_factor=0.001)], [], 0.01
    )
    assert reason == 'convergence criteria met'


def test_early_stopping_reason_points_at_the_no_symmetry_files(make_relax_result):
    reason = main._early_stopping_reason(
        [make_relax_result(r_factor=0.5)], [make_relax_result(r_factor=0.001)], 0.01
    )
    # The matching structure is not in the candidate pool, so the operator has
    # to be told where it actually landed.
    assert 'non-symmetry-constrained' in reason
    assert 'CONTCARs_no_sym' in reason
    assert 'non-sym.log' in reason


def test_early_stopping_reason_prefers_the_candidate_pool(make_relax_result):
    reason = main._early_stopping_reason(
        [make_relax_result(r_factor=0.001)], [make_relax_result(r_factor=0.001)], 0.01
    )
    assert reason == 'convergence criteria met'


def test_main_raises_on_empty_reference_pattern(
    config_factory, monkeypatch, tmp_path,
):
    config = config_factory()
    out = tmp_path / 'run'
    config['output_path'] = str(out)
    empty_ref = {'two_theta': [], 'intensity': [], 'spg_num': 225}
    _patch_main_setup(monkeypatch, config, out, empty_ref)
    monkeypatch.setattr(
        main, '_run_generation',
        lambda **kwargs: (_ for _ in ()).throw(AssertionError('should not run')),
    )

    with pytest.raises(ValueError, match='Reference pattern is empty'):
        main.main(config_path='ignored')
