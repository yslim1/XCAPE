"""Tests for utils/torchsim_utils.py (autobatcher scaling + OOM fallback).

The real torch_sim optimize path needs a GPU and a loaded MLIP, so these tests
mock ts.optimize / InFlightAutoBatcher / the calibration helpers and exercise
the pure control flow: caching, padding, retry-on-OOM, and error propagation.
"""
import pytest
import torch

from xcape.utils import torchsim_utils as tsu


@pytest.fixture(autouse=True)
def _clear_scaler_cache():
    """Isolate the module-global calibration cache between tests."""
    tsu._RAW_MAX_MEMORY_SCALER_CACHE.clear()
    yield
    tsu._RAW_MAX_MEMORY_SCALER_CACHE.clear()


class _FakeCalc:
    """Minimal stand-in for a Torch-Sim batched model."""
    memory_scales_with = "n_atoms"


class _FakeBatcher:
    """Records the max_memory_scaler it was constructed with."""
    last = None

    def __init__(self, model, memory_scales_with, max_memory_scaler):
        _FakeBatcher.last = max_memory_scaler


def _patch_ts(monkeypatch, optimize_fn):
    import torch_sim as ts
    import torch_sim.autobatching as ab
    monkeypatch.setattr(ts, "optimize", optimize_fn)
    monkeypatch.setattr(ab, "InFlightAutoBatcher", _FakeBatcher)


# =============================================================================
# is_oom_error
# =============================================================================
@pytest.mark.parametrize("msg", [
    "CUDA out of memory. Tried to allocate 2.00 GiB",
    "RuntimeError: out of memory",
    "Failed to allocate 2GB on device",
])
def test_is_oom_error_true_for_oom_messages(msg):
    assert tsu.is_oom_error(RuntimeError(msg)) is True


@pytest.mark.parametrize("msg", ["shape mismatch", "index out of range", ""])
def test_is_oom_error_false_for_non_oom(msg):
    assert tsu.is_oom_error(RuntimeError(msg)) is False


# =============================================================================
# get_max_memory_scaler
# =============================================================================
def test_get_max_memory_scaler_calibrates_once_then_caches(monkeypatch):
    import torch_sim.autobatching as ab
    calls = {"scalers": 0, "estimate": 0}

    def fake_scalers(state, mem_scales_with):
        calls["scalers"] += 1
        return "scalers-sentinel"

    def fake_estimate(state, calc, scalers):
        calls["estimate"] += 1
        return 100.0

    monkeypatch.setattr(ab, "calculate_memory_scalers", fake_scalers)
    monkeypatch.setattr(ab, "estimate_max_memory_scaler", fake_estimate)

    calc = _FakeCalc()
    first = tsu.get_max_memory_scaler(calc, init_state="s", padding=0.8)
    assert first == pytest.approx(80.0)  # raw 100 * 0.8

    # Second call reuses the cached raw value: no recalibration, and the
    # (different) padding is applied to the same raw 100.
    second = tsu.get_max_memory_scaler(calc, init_state="s", padding=0.5)
    assert second == pytest.approx(50.0)
    assert calls["estimate"] == 1
    assert calls["scalers"] == 1


def test_get_max_memory_scaler_calibrates_per_calc(monkeypatch):
    import torch_sim.autobatching as ab
    monkeypatch.setattr(ab, "calculate_memory_scalers", lambda *a, **k: None)
    seq = iter([100.0, 200.0])
    monkeypatch.setattr(ab, "estimate_max_memory_scaler", lambda *a, **k: next(seq))

    calc_a, calc_b = _FakeCalc(), _FakeCalc()
    assert tsu.get_max_memory_scaler(calc_a, "s", 1.0) == pytest.approx(100.0)
    assert tsu.get_max_memory_scaler(calc_b, "s", 1.0) == pytest.approx(200.0)


# =============================================================================
# update_cached_raw_scaler
# =============================================================================
def test_update_cached_raw_scaler_becomes_new_baseline(monkeypatch):
    import torch_sim.autobatching as ab

    def _boom(*a, **k):
        raise AssertionError("should not recalibrate after update")

    monkeypatch.setattr(ab, "calculate_memory_scalers", _boom)
    monkeypatch.setattr(ab, "estimate_max_memory_scaler", _boom)

    calc = _FakeCalc()
    # A scaler of 42 worked at padding 0.7, so the raw baseline is 42/0.7.
    tsu.update_cached_raw_scaler(calc, max_memory_scaler=42.0, padding=0.7)
    # Reading back at the same padding reproduces 42 without recalibrating.
    assert tsu.get_max_memory_scaler(calc, "s", 0.7) == pytest.approx(42.0)


# =============================================================================
# optimize_with_oom_fallback
# =============================================================================
def test_optimize_with_oom_fallback_returns_on_success(monkeypatch):
    _FakeBatcher.last = None
    _patch_ts(monkeypatch, lambda **k: "result-state")

    result, used = tsu.optimize_with_oom_fallback(
        _FakeCalc(), torch.device("cpu"), max_memory_scaler=100.0,
    )
    assert result == "result-state"
    assert used == pytest.approx(100.0)
    assert _FakeBatcher.last == pytest.approx(100.0)


def test_optimize_with_oom_fallback_shrinks_and_retries(monkeypatch):
    calls = {"n": 0}

    def flaky(**k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    _patch_ts(monkeypatch, flaky)
    result, used = tsu.optimize_with_oom_fallback(
        _FakeCalc(), torch.device("cpu"), max_memory_scaler=100.0, backoff=0.7,
    )
    assert result == "ok"
    assert calls["n"] == 2
    assert used == pytest.approx(70.0)  # 100 * 0.7 after one OOM
    assert _FakeBatcher.last == pytest.approx(70.0)


def test_optimize_with_oom_fallback_reraises_non_oom_immediately(monkeypatch):
    calls = {"n": 0}

    def boom(**k):
        calls["n"] += 1
        raise RuntimeError("shape mismatch")

    _patch_ts(monkeypatch, boom)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        tsu.optimize_with_oom_fallback(
            _FakeCalc(), torch.device("cpu"), max_memory_scaler=100.0,
        )
    assert calls["n"] == 1  # no retry on a non-OOM error


def test_optimize_with_oom_fallback_raises_when_retries_exhausted(monkeypatch):
    calls = {"n": 0}

    def always_oom(**k):
        calls["n"] += 1
        raise RuntimeError("CUDA out of memory")

    _patch_ts(monkeypatch, always_oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        tsu.optimize_with_oom_fallback(
            _FakeCalc(), torch.device("cpu"), max_memory_scaler=100.0, max_retries=2,
        )
    assert calls["n"] == 3  # initial try + 2 retries
