"""Torch-Sim autobatcher utilities for XCAPE.

This module provides helpers for calibrating and reusing the Torch-Sim
InFlightAutoBatcher's GPU memory budget (max_memory_scaler) across batch
relaxation calls, and for retrying with a smaller budget on GPU OOM.
"""
import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================
# Fraction of the empirically-probed max_memory_scaler to actually use, so the
# autobatcher keeps a safety margin instead of running right at the OOM edge.
DEFAULT_MAX_MEMORY_PADDING = 0.8

# Cache of unpadded, OOM-probed max_memory_scaler values, keyed by id(calc).
# Calibration runs real forward passes up to the OOM boundary, so it's only
# worth doing once per model rather than on every batch_relax() call.
_RAW_MAX_MEMORY_SCALER_CACHE: Dict[int, float] = {}


def get_max_memory_scaler(calc, init_state, padding: float) -> float:
    """Return a padded max_memory_scaler for calc, calibrating once and caching.

    Args:
        calc: Torch-Sim compatible batched model the scaler is calibrated for.
        init_state: Representative state used to probe the OOM boundary on
            first calibration.
        padding: Fraction of the raw probed value to actually use as the
            autobatcher's budget (e.g. 0.8 keeps 20% headroom).

    Returns:
        The padded max_memory_scaler to pass to InFlightAutoBatcher.
    """
    cache_key = id(calc)
    raw_max_memory_scaler = _RAW_MAX_MEMORY_SCALER_CACHE.get(cache_key)
    if raw_max_memory_scaler is None:
        from torch_sim.autobatching import (
            calculate_memory_scalers,
            estimate_max_memory_scaler,
        )

        scalers = calculate_memory_scalers(init_state, calc.memory_scales_with)
        raw_max_memory_scaler = estimate_max_memory_scaler(init_state, calc, scalers)
        _RAW_MAX_MEMORY_SCALER_CACHE[cache_key] = raw_max_memory_scaler
        logger.info(
            "Calibrated autobatcher max_memory_scaler=%.3g for calc=%s "
            "(cached for reuse across future batch_relax calls)",
            raw_max_memory_scaler,
            type(calc).__name__,
        )

    return raw_max_memory_scaler * padding


def update_cached_raw_scaler(calc, max_memory_scaler: float, padding: float) -> None:
    """Persist a (possibly OOM-shrunk) scaler as the new calibration baseline.

    Called after a successful (post-retry) optimize call so later stages and
    future batch_relax() calls with the same calc start from the scaler that
    actually worked, instead of re-hitting the same OOM.
    """
    _RAW_MAX_MEMORY_SCALER_CACHE[id(calc)] = max_memory_scaler / padding


def is_oom_error(exc: Exception) -> bool:
    """Return True if exc looks like a GPU out-of-memory error."""
    exc_str = str(exc)
    return any(
        msg in exc_str
        for msg in ("CUDA out of memory", "out of memory", "Failed to allocate")
    )


def optimize_with_oom_fallback(
    calc,
    device: torch.device,
    max_memory_scaler: float,
    max_retries: int = 3,
    backoff: float = 0.85,
    **optimize_kwargs: Any,
):
    """Run ts.optimize, shrinking max_memory_scaler and retrying on GPU OOM.

    The calibrated max_memory_scaler is a heuristic estimate from a probing
    pass; if a batch still OOMs at runtime (e.g. the calibration sample
    wasn't representative of the full structure_list), retry with a smaller
    autobatcher budget instead of failing the whole relaxation run.

    Args:
        calc: Torch-Sim compatible batched model to run the optimizer with.
        device: Device the states live on; used to clear the CUDA cache
            between retries.
        max_memory_scaler: Starting autobatcher memory budget to try.
        max_retries: Number of shrink-and-retry attempts before giving up.
        backoff: Factor max_memory_scaler is multiplied by after each OOM.
        **optimize_kwargs: Forwarded to ts.optimize (system, optimizer,
            convergence_fn, max_steps, etc).

    Returns:
        Tuple of (result_state, max_memory_scaler_used) so callers can carry
        a shrunk scaler into subsequent stages/calls.

    Raises:
        RuntimeError: If retries are exhausted, or if the error doesn't look
            like an OOM (re-raised immediately without retrying).
    """
    import torch_sim as ts
    from torch_sim.autobatching import InFlightAutoBatcher

    for attempt in range(max_retries + 1):
        autobatcher = InFlightAutoBatcher(
            model=calc,
            memory_scales_with=calc.memory_scales_with,
            max_memory_scaler=max_memory_scaler,
        )
        try:
            result = ts.optimize(model=calc, autobatcher=autobatcher, **optimize_kwargs)
            return result, max_memory_scaler
        except RuntimeError as exc:
            if not is_oom_error(exc) or attempt == max_retries:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            max_memory_scaler *= backoff
            logger.warning(
                "GPU OOM during ts.optimize (attempt %d/%d); retrying with "
                "max_memory_scaler shrunk to %.3g",
                attempt + 1,
                max_retries,
                max_memory_scaler,
            )
