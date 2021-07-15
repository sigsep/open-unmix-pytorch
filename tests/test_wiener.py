import pytest
import torch


from openunmix import model
from openunmix.filtering import wiener


@pytest.fixture(params=[10, 100])
def nb_frames(request):
    return int(request.param)


@pytest.fixture(params=[1, 2])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[10, 127])
def nb_bins(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def nb_sources(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def iterations(request):
    return request.param


@pytest.fixture(params=[True, False])
def softmask(request):
    return request.param


@pytest.fixture(params=[True, False])
def residual(request):
    return request.param


@pytest.fixture
def target(request, nb_frames, nb_channels, nb_bins, nb_sources):
    return torch.rand((nb_frames, nb_bins, nb_channels, nb_sources))


@pytest.fixture
def mix(request, nb_frames, nb_channels, nb_bins):
    return torch.rand((nb_frames, nb_bins, nb_channels, 2))


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def test_wiener(target, mix, iterations, softmask, residual):
    output = wiener(target, mix, iterations=iterations, softmask=softmask, residual=residual)
    # nb_frames, nb_bins, nb_channels, 2, nb_sources
    assert output.shape[:3] == mix.shape[:3]
    assert output.shape[3] == 2
    if residual:
        assert output.shape[4] == target.shape[3] + 1
    else:
        assert output.shape[4] == target.shape[3]


def test_dtype(target, mix, dtype):
    output = wiener(target.to(dtype=dtype), mix.to(dtype=dtype), iterations=1)
    assert output.dtype == dtype
