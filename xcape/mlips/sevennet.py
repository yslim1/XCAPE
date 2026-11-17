"""Factory helpers for constructing SevenNet backends from XCAPE config."""

import torch
from sevenn.sevennet_calculator import SevenNetCalculator
from sevenn.torchsim import SevenNetModel

from typing import Dict, Any


def _parse_backend_flags(backend: str) -> tuple[bool, bool]:
    """Return ``(enable_flash, enable_cueq)`` flags for a backend label."""
    if backend == 'flash':
        return True, False
    if backend == 'cueq':
        return False, True
    return False, False


def load_sevennet_calc(
    config: Dict[str, Any],
    optimizer_type: str | None = None,
) -> SevenNetCalculator | SevenNetModel:
    """Load the SevenNet backend requested by the optimizer configuration.

    Args:
        config: Configuration dictionary containing MLIP settings.
        optimizer_type: Optional optimizer override. When omitted, use the
            optimizer type from ``config``.

    Returns:
        ``SevenNetCalculator`` for sequential relaxation or ``SevenNetModel``
        for batch relaxation.
    """
    mlip_cfg = config['MLIP_setting']
    optimizer_type = optimizer_type or config['optimizer']['type']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enable_flash, enable_cueq = _parse_backend_flags(mlip_cfg.get('backend', ''))
    common_kwargs: dict[str, Any] = {
        'model': mlip_cfg['model'],
        'device': device,
        'enable_cueq': enable_cueq,
        'enable_flash': enable_flash,
    }
    modal = mlip_cfg.get('modal')
    if modal is not None:
        common_kwargs['modal'] = modal

    if optimizer_type == 'sequential':
        return SevenNetCalculator(**common_kwargs)
    if optimizer_type == 'batch':
        return SevenNetModel(**common_kwargs, dtype=torch.float32)

    raise ValueError(
        f"Unsupported optimizer type for SevenNet backend: {optimizer_type}"
    )
