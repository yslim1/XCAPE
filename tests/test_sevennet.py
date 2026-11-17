import pytest
import torch

from xcape.mlips import sevennet


class _RecordingBackend:
    """Stand-in for SevenNetCalculator/SevenNetModel that just records kwargs,
    so tests never construct/load a real SevenNet model."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _make_recording_class():
    class _Cls(_RecordingBackend):
        pass
    return _Cls


@pytest.fixture
def patched_backends(monkeypatch):
    calc_cls = _make_recording_class()
    model_cls = _make_recording_class()
    monkeypatch.setattr(sevennet, 'SevenNetCalculator', calc_cls)
    monkeypatch.setattr(sevennet, 'SevenNetModel', model_cls)
    return calc_cls, model_cls


EXPECTED_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# _parse_backend_flags
# =============================================================================
@pytest.mark.parametrize('backend,expected', [
    ('flash', (True, False)),
    ('cueq', (False, True)),
    ('', (False, False)),
    ('other', (False, False)),
])
def test_parse_backend_flags(backend, expected):
    assert sevennet._parse_backend_flags(backend) == expected


# =============================================================================
# load_sevennet_calc: sequential -> SevenNetCalculator
# =============================================================================
def test_load_sevennet_calc_sequential_from_optimizer_type_arg(patched_backends, config_factory):
    calc_cls, model_cls = patched_backends
    config = config_factory()

    result = sevennet.load_sevennet_calc(config, optimizer_type='sequential')

    assert isinstance(result, calc_cls)
    assert result.kwargs['model'] == config['MLIP_setting']['model']
    assert result.kwargs['device'].type == EXPECTED_DEVICE.type
    assert result.kwargs['modal'] == config['MLIP_setting']['modal']


def test_load_sevennet_calc_sequential_from_config_type(patched_backends, config_factory):
    calc_cls, _ = patched_backends
    config = config_factory()
    config['optimizer']['type'] = 'sequential'

    result = sevennet.load_sevennet_calc(config, optimizer_type=None)

    assert isinstance(result, calc_cls)


def test_load_sevennet_calc_sequential_omits_modal_when_absent(patched_backends, config_factory):
    calc_cls, _ = patched_backends
    config = config_factory()
    del config['MLIP_setting']['modal']

    result = sevennet.load_sevennet_calc(config, optimizer_type='sequential')

    assert 'modal' not in result.kwargs


# =============================================================================
# load_sevennet_calc: batch -> SevenNetModel
# =============================================================================
def test_load_sevennet_calc_batch_constructs_sevennet_model(patched_backends, config_factory):
    _, model_cls = patched_backends
    config = config_factory()
    config['optimizer']['type'] = 'batch'

    result = sevennet.load_sevennet_calc(config, optimizer_type=None)

    assert isinstance(result, model_cls)
    assert result.kwargs['model'] == config['MLIP_setting']['model']
    assert result.kwargs['dtype'] == torch.float32
    assert result.kwargs['device'].type == EXPECTED_DEVICE.type


# =============================================================================
# load_sevennet_calc: unsupported optimizer type
# =============================================================================
def test_load_sevennet_calc_raises_for_unsupported_optimizer_type(patched_backends, config_factory):
    config = config_factory()
    with pytest.raises(ValueError):
        sevennet.load_sevennet_calc(config, optimizer_type='parallel')


# =============================================================================
# backend flag propagation end-to-end
# =============================================================================
def test_load_sevennet_calc_propagates_flash_backend_flags(patched_backends, config_factory):
    calc_cls, _ = patched_backends
    config = config_factory()
    config['MLIP_setting']['backend'] = 'flash'

    result = sevennet.load_sevennet_calc(config, optimizer_type='sequential')

    assert result.kwargs['enable_flash'] is True
    assert result.kwargs['enable_cueq'] is False


def test_load_sevennet_calc_propagates_cueq_backend_flags(patched_backends, config_factory):
    calc_cls, _ = patched_backends
    config = config_factory()
    config['MLIP_setting']['backend'] = 'cueq'

    result = sevennet.load_sevennet_calc(config, optimizer_type='sequential')

    assert result.kwargs['enable_flash'] is False
    assert result.kwargs['enable_cueq'] is True
